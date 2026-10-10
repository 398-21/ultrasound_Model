"""
Deploy the PharmaScan API to Hugging Face (docs/DEPLOY.md has the step-by-step guide).

Two repos on your Hugging Face account:
  <user>/pharmascan-assets   private model repo: stage2_model.pt, catalogue.tar, safety.sqlite (never in git)
  <user>/pharmascan-api      public Space: only the hosted code (serving/ + src/epills/core). By default a Gradio
                             Space on ZeroGPU hardware (free for personal accounts older than 30 days; app.py =
                             serving/space_app.py, the same REST API); --docker: a Docker Space (needs HF PRO)

Commands (from the epills folder, logged in with `hf auth login` or HF_TOKEN set; the token is never printed):
  python scripts/deploy_hf.py assets --user <you>     upload the served model, catalogue and safety database,
                                                      and tag the version (for rollback)
  python scripts/deploy_hf.py space --user <you>      upload the code and set the Space's settings; the Space then
                                                      builds itself (5-10 min) and starts  [--docker]
  python scripts/deploy_hf.py feedback --user <you>   create the private dataset repo for opt-in photos (F5) and
                                                      turn collection on in the Space
  python scripts/deploy_hf.py restart --user <you>    restart the Space (loads newly uploaded assets)
"""
import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

EPILLS_ROOT = Path(__file__).resolve().parents[1]
MODEL = EPILLS_ROOT / "outputs/serving/stage2_model.pt"
CATALOGUE = EPILLS_ROOT / "data/processed/catalogue"
SAFETY = EPILLS_ROOT / "data/processed/safety/safety.sqlite"
GRADIO_VERSION = "6.30.0"  # the Space installs it; serving/space_app.py was tested with it
SPACE_README = """---
title: PharmaScan API
emoji: 💊
colorFrom: blue
colorTo: green
{sdk}
pinned: false
short_description: Pill verification and medication safety checks (prototype)
---

PharmaScan API, a non-commercial student prototype. Not a medical device: every alert defers to a pharmacist.
Interaction data: DDInter 2.0 (CC BY-NC-SA 4.0). Pill pictures: NLM C3PI (research and development use only).
Ingredients: RxNorm (NLM). Labels: openFDA. Interactive documentation: /docs.
"""


def api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))  # None: the token saved by `hf auth login`


def model_version():
    import torch
    return torch.load(MODEL, map_location="cpu", weights_only=False)["version"]


def deploy_assets(user):
    h, repo = api(), f"{user}/pharmascan-assets"
    h.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    version = model_version()
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "catalogue.tar"
        with tarfile.open(tar_path, "w") as tar:  # one file instead of ~9,000
            for f in sorted(CATALOGUE.rglob("*")):
                if f.is_file():
                    tar.add(f, arcname=f.relative_to(CATALOGUE).as_posix())
        for path, name in ((MODEL, "stage2_model.pt"), (tar_path, "catalogue.tar"), (SAFETY, "safety.sqlite")):
            print(f"uploading {name} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
            h.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo,
                          commit_message=f"{name} for model {version}")
    tag = "model-" + "".join(c if c.isalnum() or c in "-_." else "-" for c in version)
    h.create_tag(repo, tag=tag, exist_ok=True)
    print(f"done: https://huggingface.co/{repo} (private), tagged {tag}; set EPILLS_MODEL_REVISION={tag} in the "
          f"Space to pin this version, or leave it unset to always use the latest")


def deploy_space(user, docker=False):
    h, repo = api(), f"{user}/pharmascan-api"
    hardware = "cpu-basic" if docker else "zero-a10g"
    h.create_repo(repo, repo_type="space", space_sdk="docker" if docker else "gradio", space_hardware=hardware,
                  private=False, exist_ok=True)
    h.update_repo_settings(repo, repo_type="space", private=False)  # the web app calls it without a token
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        if docker:
            shutil.copy(EPILLS_ROOT / "serving/Dockerfile", stage / "Dockerfile")
            sdk = "sdk: docker\napp_port: 7860"
        else:
            shutil.copy(EPILLS_ROOT / "serving/space_app.py", stage / "app.py")
            shutil.copy(EPILLS_ROOT / "serving/requirements-space.txt", stage / "requirements.txt")
            sdk = f"sdk: gradio\nsdk_version: {GRADIO_VERSION}\npython_version: '3.12'\napp_file: app.py"
        (stage / "README.md").write_text(SPACE_README.format(sdk=sdk), encoding="utf-8")
        shutil.copytree(EPILLS_ROOT / "serving", stage / "serving",
                        ignore=shutil.ignore_patterns("__pycache__", "Dockerfile", "README.md", "space_app.py",
                                                      "requirements*.txt"))
        (stage / "src/epills").mkdir(parents=True)
        shutil.copy(EPILLS_ROOT / "src/epills/__init__.py", stage / "src/epills/__init__.py")
        shutil.copytree(EPILLS_ROOT / "src/epills/core", stage / "src/epills/core",
                        ignore=shutil.ignore_patterns("__pycache__"))
        h.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="space", commit_message="deploy API",
                        delete_patterns=["serving/**", "src/**", "Dockerfile", "app.py", "requirements.txt"])
    h.request_space_hardware(repo, hardware)
    h.add_space_variable(repo, "EPILLS_MODEL_REPO", f"{user}/pharmascan-assets")
    token = os.environ.get("HF_TOKEN")
    if token:
        h.add_space_secret(repo, "HF_TOKEN", token)  # the Space reads the private assets repo with it
    else:
        print("note: set HF_TOKEN before running this, or add a read token as the Space secret HF_TOKEN yourself "
              "(Space settings -> Variables and secrets); the assets repo is private")
    print(f"done: https://huggingface.co/spaces/{repo} builds now (5-10 min); then "
          f"https://{user.lower()}-pharmascan-api.hf.space/health and /docs")


def deploy_feedback(user):
    h = api()
    repo = f"{user}/pharmascan-feedback"
    h.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    h.add_space_variable(f"{user}/pharmascan-api", "EPILLS_FEEDBACK_REPO", repo)
    print(f"done: private dataset https://huggingface.co/datasets/{repo}; /v1/feedback is on after a restart. "
          f"The Space's HF_TOKEN must have write access for it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["assets", "space", "feedback", "restart"])
    parser.add_argument("--user", required=True, help="your Hugging Face user (or organisation) name")
    parser.add_argument("--docker", action="store_true", help="space: a Docker Space instead (needs HF PRO)")
    args = parser.parse_args()
    if args.command == "assets":
        for p in (MODEL, CATALOGUE / "catalogue.sqlite", SAFETY):
            if not p.exists():
                sys.exit(f"missing {p}: run export_model.py / build_catalogue.py / build_safety_db.py first")
        deploy_assets(args.user)
    elif args.command == "space":
        deploy_space(args.user, args.docker)
    elif args.command == "feedback":
        deploy_feedback(args.user)
    else:
        api().restart_space(f"{args.user}/pharmascan-api")
        print("restarting; it reloads the assets in about a minute")


if __name__ == "__main__":
    main()
