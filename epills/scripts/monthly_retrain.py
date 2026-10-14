"""
Monthly retraining on the verified pill photos users chose to share (docs/features.md F5, docs/DEPLOY.md).

Runs on the desktop GPU (the free Space has none). Steps, run in order (each can be re-run on its own):
  pull     download the photos from the private dataset repo (EPILLS_FEEDBACK_REPO) -> data/feedback/<month>/
  review   list the user-confirmed labels waiting for review; --approve / --reject item ids (blueprint B15)
  build    curate: pharmacist-checked labels (and approved user-confirmed ones with --include-user-confirmed); a known
           medicine (NDC9 in our groups) joins its group, a new one becomes a new class; exact duplicates dropped; the newest
           20% of each medicine's photos held out as this month's feedback test. Writes the manifest and the config.
  train    fine-tune the served run on it (its weights and ArcFace table; phase 2, gentle learning rate 1e-5)
  check    calibrate (crop window on validation), run the same test sets as the served model plus this month's
           feedback test, and apply the release gate -> outputs/runs/<name>/release_report.md (PASS / FAIL)
  publish  only after a person has read the report: export the model, rebuild the catalogue, upload a new tagged
           version to the assets repo and restart the Space (scripts/deploy_hf.py)

The gate (vs the served model, same measurements): validation correct matches may drop at most 2 points, wrong
matches may rise at most 0.2 points, unknown pills rejected stays >= 94% (or, where the live model is below that,
drops at most 1 point); on every test set both runs have, the same. Validation and test splits never take feedback photos, so the comparison stays fair.

Usage, from the epills folder:
  python scripts/monthly_retrain.py pull build train check     # from the live run (outputs/serving/served_run.txt)
  python scripts/monthly_retrain.py publish --from-run outputs/runs/stage2_monthly_<YYYYMM> --user <hf user>
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EPILLS_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MONTH = datetime.date.today().strftime("%Y-%m")
CROPS = EPILLS_ROOT / "data/interim/c3pi/crops/images"
GATE = {"correct_drop": 0.02, "wrong_rise": 0.002, "unknown_drop": 0.01, "min_unknown_rejected": 0.94}
SERVED = EPILLS_ROOT / "outputs/serving/served_run.txt"  # the run behind the exported model (export_model.py)


def run(*cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([PY, *map(str, cmd)], cwd=EPILLS_ROOT, check=True)


def paths(month):
    name = f"stage2_monthly_{month.replace('-', '')}"
    return {"raw": EPILLS_ROOT / "data/feedback" / month,
            "crops": CROPS / "feedback" / month,
            "manifest": EPILLS_ROOT / f"data/processed/c3pi_splits/manifest_monthly_{month}.csv",
            "config": EPILLS_ROOT / f"configs/{name}.yaml",
            "name": name, "run": EPILLS_ROOT / "outputs/runs" / name}


def pull(p):
    from huggingface_hub import snapshot_download
    repo = os.environ.get("EPILLS_FEEDBACK_REPO") or sys.exit("set EPILLS_FEEDBACK_REPO (e.g. <user>/pharmascan-feedback)")
    snapshot_download(repo, repo_type="dataset", local_dir=p["raw"], token=os.environ.get("HF_TOKEN"))
    print(f"pulled {sum(1 for _ in p['raw'].rglob('*.jpg')):,} photos -> {p['raw']}")


def reviews_of(p):
    f = p["raw"] / "reviews.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def review(p, approve, reject):
    """User-confirmed labels wait for a person's review (blueprint B15): list them, or record decisions."""
    decisions = reviews_of(p)
    for items, decision in ((approve, "approved"), (reject, "rejected")):
        for item in filter(None, (items or "").split(",")):
            decisions[item.strip()] = {"decision": decision,
                                       "at": datetime.datetime.now().isoformat(timespec="seconds")}
    (p["raw"] / "reviews.json").write_text(json.dumps(decisions, indent=1), encoding="utf-8")
    pending = [m for m in (json.loads(f.read_text(encoding="utf-8")) for f in sorted(p["raw"].rglob("*.json"))
                           if f.name != "reviews.json")
               if m.get("label_source") == "user_confirmed" and m.get("item") not in decisions]
    for m in pending:
        print(f"{m['item']}  {m['medication_name']!r}  app said {m.get('app_decision')}  "
              f"photo {next(p['raw'].rglob(m['item'] + '.jpg'), '?')}")
    print(f"{len(pending)} pending; {sum(d['decision'] == 'approved' for d in decisions.values())} approved, "
          f"{sum(d['decision'] == 'rejected' for d in decisions.values())} rejected")


def build(p, served, include_user):
    items, approved = [], {k for k, d in reviews_of(p).items() if d["decision"] == "approved"}
    for meta_file in sorted(p["raw"].rglob("*.json")):
        if meta_file.name == "reviews.json":
            continue
        m = json.loads(meta_file.read_text(encoding="utf-8"))
        jpg = meta_file.with_suffix(".jpg")
        user_ok = include_user and m["label_source"] == "user_confirmed" and m.get("item") in approved
        if jpg.exists() and (m["label_source"] == "pharmacist" or user_ok):
            items.append({**m, "jpg": jpg, "sha1": hashlib.sha1(jpg.read_bytes()).hexdigest()})
    if not items:
        sys.exit("no usable feedback photos this month")
    fb = pd.DataFrame(items).drop_duplicates("sha1")
    import torch
    served_cfg = torch.load(served / "best.pt", map_location="cpu", weights_only=False)["config"]
    base = pd.read_csv(EPILLS_ROOT / served_cfg["data"]["manifest"], dtype=str, keep_default_na=False)
    groups = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/groups.csv", dtype=str, keep_default_na=False)
    group_of_ndc = groups[groups.split == "train"].groupby("ndc9").group.first()
    held_out = set(groups[groups.split != "train"].ndc9)

    def ndc9(x):
        d = re.sub(r"\D", "", str(x or ""))
        return d[:9] if len(d) >= 9 else ""

    def label(r):
        n = ndc9(r.ndc)
        if n in held_out:
            return None  # a validation or test medicine: never trained on, so comparisons stay fair
        if n in group_of_ndc.index:
            return group_of_ndc[n]
        return "fb:" + (n or re.sub(r"[^a-z0-9]+", "-", r.medication_name.lower()).strip("-"))

    fb["label"] = fb.apply(label, axis=1)
    skipped = int(fb.label.isna().sum())
    fb = fb[fb.label.notna()].sort_values("received_at")
    fb["rank"] = fb.groupby("label").cumcount(ascending=False)  # 0 = newest
    fb["n"] = fb.groupby("label").label.transform("size")
    fb["split"] = np.where((fb.n >= 4) & (fb["rank"] < (fb.n * 0.2).clip(lower=1)), "feedback_test", "train")
    p["crops"].mkdir(parents=True, exist_ok=True)
    rows = []
    for r in fb.itertuples():
        dest = p["crops"] / r.jpg.name
        dest.write_bytes(r.jpg.read_bytes())
        rows.append({"crop_path": dest.relative_to(CROPS).as_posix(), "split": r.split, "label": r.label,
                     "group": r.label, "ndc9": ndc9(r.ndc), "proprietary_name": r.medication_name,
                     "shape": "", "color": "", "source": "feedback"})
    manifest = pd.concat([base, pd.DataFrame(rows).reindex(columns=base.columns.union(["source"]), fill_value="")],
                         ignore_index=True)
    manifest.to_csv(p["manifest"], index=False)
    cfg = dict(served_cfg)
    cfg.update(name=p["name"], init_from=str((served / "best.pt").relative_to(EPILLS_ROOT)).replace("\\", "/"))
    cfg["data"] = {**cfg["data"], "manifest": str(p["manifest"].relative_to(EPILLS_ROOT)).replace("\\", "/")}
    cfg["loss"] = {**cfg["loss"], "arcface_margin_warmup_epochs": 0}
    cfg["phases"] = [{"phase": 2, "epochs": 10, "p": 16, "k": 4, "lr": 1.0e-5, "head_lr": 1.0e-5, "metric_lr": 1.0e-4}]
    cfg["early_stopping_patience"] = 4
    cfg["select_metric"] = served_cfg.get("select_metric", "top1_all")
    p["config"].write_text(f"# Monthly retraining {MONTH}, made by scripts/monthly_retrain.py from {served.name}\n"
                           + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"{len(fb):,} feedback photos of {fb.label.nunique()} medicines ({fb.label.str.startswith('fb:').sum():,} "
          f"photos of new medicines; {(fb.split == 'feedback_test').sum():,} held out); {skipped} skipped as "
          f"validation/test medicines -> {p['manifest'].name}, {p['config'].name}")


def feedback_test(run_dir, p):
    """Closed-set top-1 on this month's held-out feedback photos, the medicines enrolled from their other photos."""
    import torch
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(EPILLS_ROOT / "src"))
    from epills.core.models.hybrid import HybridPillNet
    from epills.training.datasets.pills import PillCrops, Vocab, eval_transform
    from epills.training.eval.retrieval import embed
    m = pd.read_csv(p["manifest"], dtype=str, keep_default_na=False)
    fb = m[m.source == "feedback"].reset_index(drop=True)
    if not (fb.split == "feedback_test").any():
        return None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = HybridPillNet.from_checkpoint(run_dir / "best.pt", dev)
    emb, lab = embed(model, DataLoader(PillCrops(fb, CROPS, eval_transform(), Vocab(m[m.split == "train"])),
                                       batch_size=64), dev)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    test = (fb.split == "feedback_test").to_numpy()
    labels = np.unique(lab[~test])
    protos = np.stack([emb[~test & (lab == g)].mean(0) for g in labels])
    pred = labels[(emb[test] @ protos.T).argmax(1)]
    return float((pred == lab[test]).mean())


def check(p, served):
    new = p["run"]
    run("scripts/calibrate_stage2.py", "--run", new, "--windows", "1.3", "1.4", "1.55", "1.7", "1.85", "2.0")
    sets = [s for s in ("c3pi_holdout", "epillid", "mediseg") if (served / "eval" / f"{s}.json").exists()]
    run("scripts/evaluate_stage2.py", "--run", new, "--sets", *sets)
    old_cal, new_cal = (json.loads((r / "calibration.json").read_text())["operating_point"] for r in (served, new))
    rows, ok = [], True

    def gate(name, a, b):
        nonlocal ok
        good = (b["correct_match"] >= a["correct_match"] - GATE["correct_drop"]
                and b["wrong_match"] <= a["wrong_match"] + GATE["wrong_rise"]
                and b["unknown_rejected"] >= min(GATE["min_unknown_rejected"],
                                                 a["unknown_rejected"] - GATE["unknown_drop"]))
        ok &= good
        rows.append(f"| {name} | {a['correct_match']:.1%} → {b['correct_match']:.1%} | {a['wrong_match']:.2%} → "
                    f"{b['wrong_match']:.2%} | {a['unknown_rejected']:.1%} → {b['unknown_rejected']:.1%} | "
                    f"{'pass' if good else '**FAIL**'} |")

    gate("validation (calibration)", old_cal, new_cal)
    for s in ("c3pi_holdout", "mediseg"):
        if s in sets:
            a, b = (json.loads((r / "eval" / f"{s}.json").read_text())["open_set"]["gallery_10"]["at_calibration"]
                    for r in (served, new))
            gate(s, a, b)
    fb_old, fb_new = feedback_test(served, p), feedback_test(new, p)
    verdict = "PASS: may be published after a person has read this report" if ok else "FAIL: do not publish"
    report = [f"# Release report: {p['name']} ({MONTH})", "",
              f"Fine-tuned from `{served.name}` on this month's verified feedback photos. **{verdict}.**", "",
              "| Test (10 medications, 3 enrolment photos) | Correct matches | Wrong matches | Unknown pills rejected | Gate |",
              "|---|---|---|---|---|", *rows, "",
              f"This month's held-out feedback photos, top-1 among the feedback medicines: "
              f"{'n/a' if fb_old is None else f'{fb_old:.1%} → {fb_new:.1%}'}", "",
              "Before publishing, a person checks this report (and, for a registered device, the change-management "
              "records; docs/features.md F5). Publish: `python scripts/monthly_retrain.py publish --from-run "
              f"outputs/runs/{p['name']} --user <hf user>`."]
    (new / "release_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


def publish(new_run, user):
    run("scripts/export_model.py", "--run", new_run)
    run("scripts/build_catalogue.py")
    run("scripts/deploy_hf.py", "assets", "--user", user)
    run("scripts/deploy_hf.py", "restart", "--user", user)
    SERVED.write_text(str(new_run.relative_to(EPILLS_ROOT)).replace("\\", "/"), encoding="utf-8")
    print(f"published; {SERVED.name} now points to {new_run.name}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the report has arrows; logs on Windows are cp1252
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=["pull", "review", "build", "train", "check", "publish"])
    parser.add_argument("--from-run", type=Path,
                        help="the served run (default: outputs/serving/served_run.txt); for publish: the approved run")
    parser.add_argument("--month", default=MONTH)
    parser.add_argument("--user", help="Hugging Face user (publish)")
    parser.add_argument("--include-user-confirmed", action="store_true",
                        help="also train on labels the user confirmed and a person approved (review)")
    parser.add_argument("--approve", help="review: comma-separated item ids to approve")
    parser.add_argument("--reject", help="review: comma-separated item ids to reject")
    parser.add_argument("--smoke", action="store_true", help="train for a few batches only (testing the pipeline)")
    args = parser.parse_args()
    if args.from_run is None:
        if not SERVED.exists():
            sys.exit(f"give --from-run (no {SERVED} yet)")
        args.from_run = Path(SERVED.read_text(encoding="utf-8").strip())
    served = args.from_run if args.from_run.is_absolute() else EPILLS_ROOT / args.from_run
    p = paths(args.month)
    if "publish" in args.steps:
        report = (served / "release_report.md")
        if not report.exists() or "PASS" not in report.read_text(encoding="utf-8").splitlines()[2]:
            sys.exit(f"{report} missing or not PASS: not publishing")
        publish(served, args.user or sys.exit("--user is required to publish"))
        return
    if "pull" in args.steps:
        pull(p)
    if "review" in args.steps:
        review(p, args.approve, args.reject)
    if "build" in args.steps:
        build(p, served, args.include_user_confirmed)
    if "train" in args.steps:
        extra = ["--epochs", "1", "--max-batches", "20"] if args.smoke else []
        run("scripts/train_stage2.py", "--config", p["config"], *extra)
    if "check" in args.steps:
        check(p, served)


if __name__ == "__main__":
    main()
