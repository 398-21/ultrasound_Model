"""
Opt-in collection of verified pill photos for the monthly retraining (docs/features.md F5, docs/DEPLOY.md).

Off unless configured:
  EPILLS_FEEDBACK_REPO   a *private* Hugging Face dataset repo; photos are pushed to it every 10 minutes
                         (huggingface_hub CommitScheduler; needs HF_TOKEN with write access)
  EPILLS_FEEDBACK_DIR    local folder (default /tmp/pharmascan_feedback); alone, it keeps photos locally (testing)

Only what retraining needs is stored: the pill crop (re-encoded, so no EXIF or location), the medication label, how
the label was confirmed (pharmacist / user), the app's decision, versions, and the app's consent id. Photos are
grouped by consent id, so DELETE /v1/feedback/{consent_id} removes all of a person's photos when they withdraw
consent (PDPA). Photos never feed the model directly: the monthly job curates them and a person approves the
new model (scripts/monthly_retrain.py).
"""
import json
import re
import shutil
import threading
import uuid
from pathlib import Path


def safe_id(text):
    """A consent id usable as a folder name (letters, digits, - and _ only)."""
    cid = re.sub(r"[^A-Za-z0-9_-]", "", str(text))[:64]
    if not cid:
        raise ValueError("consent_id must contain letters or digits")
    return cid


class FeedbackStore:
    def __init__(self, folder, repo_id=None, token=None, every_minutes=10):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.repo_id, self.token = repo_id, token
        self.lock = threading.Lock()
        self.scheduler = None
        if repo_id:
            from huggingface_hub import CommitScheduler
            self.scheduler = CommitScheduler(repo_id=repo_id, repo_type="dataset", folder_path=self.folder,
                                             path_in_repo="feedback", every=every_minutes, private=True, token=token)
            self.lock = self.scheduler.lock  # never upload a half-written item

    def save(self, crop, meta):
        cid, item = safe_id(meta["consent_id"]), uuid.uuid4().hex
        with self.lock:
            folder = self.folder / cid
            folder.mkdir(parents=True, exist_ok=True)
            crop.convert("RGB").save(folder / f"{item}.jpg", quality=92)  # re-encoded: no EXIF
            (folder / f"{item}.json").write_text(json.dumps({**meta, "item": item}, indent=1), encoding="utf-8")
        return item

    def delete(self, consent_id):
        cid = safe_id(consent_id)
        with self.lock:
            local = self.folder / cid
            removed = sum(1 for p in local.glob("*.jpg")) if local.exists() else 0
            shutil.rmtree(local, ignore_errors=True)
        remote = False
        if self.repo_id:
            from huggingface_hub import HfApi
            from huggingface_hub.utils import EntryNotFoundError
            try:
                HfApi(token=self.token).delete_folder(path_in_repo=f"feedback/{cid}", repo_id=self.repo_id,
                                                      repo_type="dataset",
                                                      commit_message=f"withdrawal of consent {cid}")
                remote = True
            except EntryNotFoundError:
                pass
        return {"local_photos_removed": removed, "removed_from_repo": remote}
