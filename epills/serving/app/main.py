"""
PharmaScan API: Stage 2 (pill verification and identification) and Stage 3 (safety checks) for the app.

Stateless by design: no photo, gallery or schedule is stored on the server. The app keeps each patient's gallery
embeddings and schedule on the phone and sends what a request needs. The endpoints are described in
serving/README.md; the features behind them in docs/features.md.

Assets (not in git), each from an environment variable with a local path, else from the Hugging Face repo in
EPILLS_MODEL_REPO (HF_TOKEN if private), else the local build output:
  EPILLS_SAFETY_DB   safety.sqlite                  data/processed/safety/safety.sqlite
  EPILLS_MODEL       stage2_model.pt                outputs/serving/stage2_model.pt   (scripts/export_model.py)
  EPILLS_CATALOGUE   catalogue.tar (or a folder)    data/processed/catalogue          (scripts/build_catalogue.py)
EPILLS_MODEL_REVISION pins a version of that repo (a tag made by scripts/deploy_hf.py); unset: the latest.
EPILLS_PRELOAD=1 loads them all at start-up (the Docker image sets it); otherwise on first use.
EPILLS_DEVICE: cpu (default) or cuda.
EPILLS_CORS_ORIGINS: comma-separated web origins allowed to call the API from a browser (default * , any).
EPILLS_API_KEY: if set, every /v1 request needs the header X-API-Key with this value.
EPILLS_FEEDBACK_REPO / EPILLS_FEEDBACK_DIR: opt-in photo collection for retraining (serving/app/feedback.py).

Run locally, from the epills folder:
    uvicorn serving.app.main:app --reload --port 7860
"""
import datetime
import io
import json
import os
import tarfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from epills.core.inference.matching import decide
from epills.core.inference.preprocess import WINDOW, crop_window
from epills.core.safety import openfda
from epills.core.safety.db import SafetyDB
from epills.core.safety.rules import check_dose, check_medications, resolve_medication
from epills.core.safety.rxnorm import RxNormClient

API_VERSION = "0.4.0"
LOCAL = {"safety_db": Path("data/processed/safety/safety.sqlite"),
         "embedder": Path("outputs/serving/stage2_model.pt"),
         "catalogue": Path("data/processed/catalogue")}
IDENTIFY_NOTE = ("Possible matches from a catalogue of US pills, ranked by how alike they look. A photo can't confirm "
                 "what a pill is. Check with your pharmacist.")
MATCH_MESSAGES = {
    "match": "This is your {name}.",
    "unknown": "This pill isn't one of your medicines. Set it aside. Check with your pharmacist.",
    "ambiguous": "This pill looks like more than one of your medicines. Turn it over and photograph it again.",
}


def _asset(env, repo_path, local):
    if os.environ.get(env):
        return Path(os.environ[env])
    repo = os.environ.get("EPILLS_MODEL_REPO")
    if repo:
        from huggingface_hub import hf_hub_download
        # EPILLS_MODEL_REVISION pins a released version (a tag such as model-stage2_run4-2026-10-09): rollback
        return Path(hf_hub_download(repo, repo_path, revision=os.environ.get("EPILLS_MODEL_REVISION"),
                                    token=os.environ.get("HF_TOKEN")))
    return local


def _unpacked(path):
    """A catalogue: its folder, or a .tar of it (one download instead of ~9,000 files), unpacked next to it once."""
    path = Path(path)
    if path.suffix != ".tar":
        return path
    folder = path.with_name(path.stem + "_unpacked")
    if not (folder / "catalogue.sqlite").exists():
        with tarfile.open(path) as tar:
            tar.extractall(folder, filter="data")
    return folder


def _make(name):
    if name == "safety_db":
        return SafetyDB(_asset("EPILLS_SAFETY_DB", "safety.sqlite", LOCAL[name]))
    if name == "embedder":
        from epills.core.inference.embedder import Embedder  # torch: imported only when the model is needed
        return Embedder(_asset("EPILLS_MODEL", "stage2_model.pt", LOCAL[name]),
                        device=os.environ.get("EPILLS_DEVICE", "cpu"))
    if name == "catalogue":
        from epills.core.catalogue.store import Catalogue
        return Catalogue(_unpacked(_asset("EPILLS_CATALOGUE", "catalogue.tar", LOCAL[name])))
    if name == "feedback":
        from serving.app.feedback import FeedbackStore
        repo, folder = os.environ.get("EPILLS_FEEDBACK_REPO"), os.environ.get("EPILLS_FEEDBACK_DIR")
        if not repo and not folder:
            raise RuntimeError("feedback collection is off (set EPILLS_FEEDBACK_REPO)")
        return FeedbackStore(folder or "/tmp/pharmascan_feedback", repo, os.environ.get("HF_TOKEN"))
    raise KeyError(name)


_lock = threading.Lock()
_errors = {}


def asset(name):
    """The asset, loaded once; 503 with the reason if it can't be loaded (the rest of the API keeps working)."""
    if getattr(app.state, name) is None:
        with _lock:
            if getattr(app.state, name) is None:
                try:
                    setattr(app.state, name, _make(name))
                    _errors.pop(name, None)
                except Exception as error:  # report, don't crash the whole API
                    _errors[name] = f"{type(error).__name__}: {error}"
    if getattr(app.state, name) is None:
        raise HTTPException(503, f"{name} not available: {_errors.get(name)}")
    return getattr(app.state, name)


def catalogue_for_identify():
    """The catalogue, checked against the served model: embeddings of another model version can't be compared."""
    cat, emb = asset("catalogue"), asset("embedder")
    if cat.model_version != emb.version:
        raise HTTPException(503, f"catalogue built for model {cat.model_version}, but {emb.version} is served; "
                                 f"rebuild it with scripts/build_catalogue.py")
    return cat


@asynccontextmanager
async def lifespan(_app):
    if os.environ.get("EPILLS_PRELOAD") == "1":
        def preload():
            for name in ("safety_db", "embedder", "catalogue"):
                try:
                    asset(name)
                except HTTPException:
                    pass
        threading.Thread(target=preload, daemon=True).start()
    yield


app = FastAPI(
    title="PharmaScan API",
    version=API_VERSION,
    lifespan=lifespan,
    description="Pill verification against a patient's enrolled medications, pill identification and pictures from "
                "a catalogue of US pills, and medication safety checks. A medication verification aid, not a medical "
                "device: every alert defers to a pharmacist. Interaction data: DDInter 2.0 (CC BY-NC-SA 4.0); "
                "ingredients: RxNorm (NLM); labels: openFDA; pill pictures: NLM C3PI (research and development use).",
)
app.state.rxnorm = RxNormClient()
app.state.safety_db = app.state.embedder = app.state.catalogue = app.state.feedback = None
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("EPILLS_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def api_key(request: Request, call_next):
    """Optional shared key (EPILLS_API_KEY) on /v1 routes; /health and /docs stay open."""
    key = os.environ.get("EPILLS_API_KEY")
    if key and request.url.path.startswith("/v1") and request.method != "OPTIONS" \
            and request.headers.get("X-API-Key") != key:
        return JSONResponse({"detail": "missing or wrong X-API-Key"}, status_code=401)
    return await call_next(request)


# ---- request models ---------------------------------------------------------------------------------------------
class Medication(BaseModel):
    id: str = Field(description="The app's id for this medication")
    name: str | None = Field(None, description="Name as on the label, brand or generic, e.g. 'Lipitor 10 mg'")
    rxcui: str | None = Field(None, description="RxNorm code, if the app has it")
    ndc: str | None = Field(None, description="National Drug Code from a US bottle, e.g. '00071-0155-23'")
    ingredients: list[str] | None = Field(None, description="Active ingredients from the label; use this for "
                                                            "medicines without US codes")


class MedicationsRequest(BaseModel):
    medications: list[Medication] = Field(min_length=1)


class ScheduledMedication(BaseModel):
    medication_id: str
    count: int = Field(1, ge=0, description="Pills of this medication expected in this dose")


class CheckedPill(BaseModel):
    medication_id: str | None = Field(None, description="Best match from /v1/match (None if unknown)")
    decision: Literal["match", "unknown", "ambiguous"]


class DoseRequest(BaseModel):
    schedule: list[ScheduledMedication]
    pills: list[CheckedPill]
    names: dict[str, str] = Field(default_factory=dict, description="medication_id -> name, for messages")


class GalleryMedication(BaseModel):
    medication_id: str
    name: str | None = Field(None, description="Shown in the answer, e.g. 'Metformin 500 mg'")
    embeddings: list[list[float]] = Field(min_length=1, description="From /v1/embed, one per enrolment photo")


class MatchRequest(BaseModel):
    queries: list[list[float]] = Field(min_length=1, description="Embeddings of the pills to check (/v1/embed)")
    gallery: list[GalleryMedication] = Field(min_length=1, description="The patient's enrolled medications")
    model_version: str | None = Field(None, description="Version the gallery was embedded with; checked if given")


# ---- helpers ------------------------------------------------------------------------------------------------------
async def read_image(upload):
    try:
        image = Image.open(io.BytesIO(await upload.read()))
        return ImageOps.exif_transpose(image).convert("RGB")  # phone photos carry their rotation in EXIF
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(422, f"{upload.filename or 'image'} is not a readable image: {error}")


def parse_json(text, what):
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError as error:
        raise HTTPException(422, f"{what} is not valid JSON: {error}")


def parse_boxes(text, n=None):
    """'[[x0, y0, x1, y1], ...]' -> list of 4-tuples (photo pixels)."""
    boxes = parse_json(text, "boxes")
    if boxes is None:
        return None
    if not isinstance(boxes, list) or not all(isinstance(b, list) and len(b) == 4 for b in boxes):
        raise HTTPException(422, "boxes must be a list of [x0, y0, x1, y1]")
    if any(b[2] <= b[0] or b[3] <= b[1] for b in boxes):
        raise HTTPException(422, "every box needs x1 > x0 and y1 > y0")
    if n is not None and len(boxes) != n:
        raise HTTPException(422, f"{len(boxes)} boxes for {n} images")
    return [tuple(float(v) for v in b) for b in boxes]


def parse_box(text):
    """'[x0, y0, x1, y1]' or 'x0,y0,x1,y1' -> a 4-tuple, or None."""
    if not text or not text.strip():
        return None
    t = text.strip()
    values = parse_json(t, "box") if t.startswith("[") else t.split(",")
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        raise HTTPException(422, "box must be 4 numbers: x0, y0, x1, y1")
    return parse_boxes(json.dumps([values]))[0]


def match_decisions(queries, gallery, embedder):
    dim = queries.shape[1]
    for g in gallery:
        if any(len(e) != dim for e in g.embeddings):
            raise HTTPException(422, f"gallery embeddings for {g.medication_id} don't have {dim} numbers")
    names = {g.medication_id: g.name or g.medication_id for g in gallery}
    cal = embedder.calibration
    out = []
    for d in decide(queries, {g.medication_id: np.asarray(g.embeddings, np.float32) for g in gallery},
                    cal["threshold"], cal["margin"]):
        matched = d.decision == "match"
        out.append({
            "decision": d.decision,
            "medication_id": d.medication_id if matched else None,
            "name": names[d.medication_id] if matched else None,
            "similarity": round(d.similarity, 4),
            "closest": {"medication_id": d.medication_id, "name": names[d.medication_id],
                        "similarity": round(d.similarity, 4)},
            "runner_up": ({"medication_id": d.runner_up_id, "name": names[d.runner_up_id],
                           "similarity": round(d.runner_up_similarity, 4)} if d.runner_up_id is not None else None),
            "message": MATCH_MESSAGES[d.decision].format(name=names[d.medication_id]),
        })
    return out


def product_summary(cat, pill_id, **extra):
    p = cat.product(pill_id)
    return {
        "pill_id": pill_id, "name": p["name"], "rxnorm_name": p["rxnorm_name"], "generic_name": p["generic_name"],
        "manufacturer": p["manufacturer"], "ingredients": [i["name"] for i in p["ingredients"]],
        "shape": p["shape"], "colour": p["colour"], "size_mm": p["size_mm"], "imprint": p["imprint"],
        "images": [{"url": f"/v1/pills/{pill_id}/image/{i['n']}", "kind": i["kind"]} for i in p["images"]],
        **extra,
    }


def identify_pill(embeddings, imprint=None, shape=None, colour=None, top_k=5):
    cat = catalogue_for_identify()
    candidates, notes = cat.identify(embeddings, top_k=top_k, imprint=imprint, shape=shape, colour=colour)
    return {"candidates": [product_summary(cat, c.pop("pill_id"), **c) for c in candidates], "notes": notes,
            "note": IDENTIFY_NOTE}


# ---- endpoints: health ----------------------------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness check, and which assets are loaded (the model and catalogue load on first use)."""
    try:
        db = asset("safety_db")
        safety = {"loaded": True, "built_at": db.meta.get("built_at"), "pairs": db.meta.get("pairs")}
    except HTTPException as error:
        safety = {"loaded": False, "error": error.detail}
    emb, cat = app.state.embedder, app.state.catalogue
    # where photos run: the Space's GpuOrCpu forward knows (zerogpu, or cpu after a refusal), else the model's device
    device = getattr(getattr(emb, "forward", None), "device", None) or getattr(emb, "device", None)
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "model_repo": os.environ.get("EPILLS_MODEL_REPO", "not set"),
        "model": {"loaded": emb is not None, "version": getattr(emb, "version", None),
                  "device": str(device) if device else None, "error": _errors.get("embedder")},
        "catalogue": {"loaded": cat is not None, "products": len(cat) if cat else None,
                      "model_version": getattr(cat, "model_version", None), "error": _errors.get("catalogue")},
        "safety_db": safety,
    }


# ---- endpoints: Stage 2 -------------------------------------------------------------------------------------------
@app.post("/v1/embed")
async def embed(image: UploadFile = File(description="A pill crop, or a photo with `box`"),
                box: str | None = Form(None, description="Optional [x0, y0, x1, y1] of the pill in the photo; the "
                                                         "server then crops it with the crop contract")):
    """
    A pill photo -> its 256-number embedding. Used at **enrolment** (the app stores the embeddings as the patient's
    gallery, with the model version) and before `/v1/match`. Without a box, the image must already be framed like a
    crop: the pill in the middle, its long side about 70% of the image.
    """
    emb = asset("embedder")
    photo = await read_image(image)
    b = parse_box(box)
    vector = (await run_in_threadpool(emb.embed_photo, photo, [b] if b else None))[0]
    return {"embedding": [round(float(v), 6) for v in vector], "dim": int(vector.shape[0]),
            "model_version": emb.version, "framed_by": "box" if b else "as sent"}


@app.post("/v1/match")
def match(request: MatchRequest):
    """
    **Which of my medicines is this?** Pill embeddings + the patient's gallery -> per pill: `match` (with the
    medication's name), `unknown` (not one of theirs: set it aside) or `ambiguous` (turn it over, photograph again).
    The threshold and margin are the served model's calibration.
    """
    emb = asset("embedder")
    if request.model_version and request.model_version != emb.version:
        raise HTTPException(409, f"gallery embedded with model {request.model_version}, but {emb.version} is "
                                 f"served: re-embed the enrolment photos with /v1/embed")
    queries = np.asarray(request.queries, np.float32)
    return {"results": match_decisions(queries, request.gallery, emb), "model_version": emb.version,
            "threshold": emb.calibration["threshold"], "margin": emb.calibration["margin"]}


@app.post("/v1/identify")
async def identify(images: list[UploadFile] = File(description="One pill: 1 photo, or 2 (front and back)"),
                   boxes: str | None = Form(None, description="Optional [[x0, y0, x1, y1], ...], one per image"),
                   imprint: str | None = Form(None, description="Letters or numbers on the pill, e.g. 'M 547'"),
                   shape: str | None = Form(None, description="e.g. ROUND, OVAL, CAPSULE"),
                   colour: str | None = Form(None, description="e.g. WHITE, PINK"),
                   top_k: int = Form(5, ge=1, le=20)):
    """
    **What is this pill?** Photo(s) of one pill -> the most alike pills in the catalogue, with names and pictures.
    Typing the imprint narrows the list a lot. Always possible matches, never a confirmed identity.
    """
    if not 1 <= len(images) <= 2:
        raise HTTPException(422, "send 1 or 2 photos of the same pill")
    emb = asset("embedder")
    photos = [await read_image(i) for i in images]
    box_list = parse_boxes(boxes, len(photos))
    vectors = np.concatenate([await run_in_threadpool(emb.embed_photo, p, [box_list[i]] if box_list else None)
                              for i, p in enumerate(photos)])
    return {**identify_pill(vectors, imprint, shape, colour, top_k), "model_version": emb.version}


@app.post("/v1/scan")
async def scan(image: UploadFile = File(description="A full photo of the pills"),
               boxes: str | None = Form(None, description="Pill boxes from a detector, [[x0, y0, x1, y1], ...]; "
                                                          "without them a stand-in finder looks for pills on a "
                                                          "plain background"),
               gallery: str | None = Form(None, description="JSON list of {medication_id, name, embeddings}: the "
                                                            "patient's gallery. Without it, each pill is "
                                                            "identified against the catalogue"),
               schedule: str | None = Form(None, description="JSON list of {medication_id, count} for this dose: "
                                                             "adds the dose check"),
               model_version: str | None = Form(None, description="Version the gallery was embedded with"),
               top_k: int = Form(3, ge=1, le=10)):
    """
    **One call for a dose check:** full photo -> every pill's box and decision (with the patient's gallery), or its
    catalogue candidates (without one); with a schedule, also the dose-check alerts of `/v1/safety/dose`.
    """
    emb = asset("embedder")
    photo = await read_image(image)
    box_list = parse_boxes(boxes)
    found_by = "boxes sent"
    if box_list is None:
        from epills.core.inference.pill_finder import find_pills
        box_list, found_by = find_pills(photo), "stand-in finder (plain backgrounds only)"
    result = {"found_by": found_by, "model_version": emb.version, "pills": []}
    if not box_list:
        result["message"] = ("No pill found. Spread the pills apart on a plain background and photograph them "
                             "again.")
        return result
    vectors = await run_in_threadpool(emb.embed_photo, photo, box_list)  # a GPU queue wait mustn't block the API

    if gallery:
        try:
            meds = [GalleryMedication(**g) for g in parse_json(gallery, "gallery")]
        except (TypeError, ValidationError) as error:
            raise HTTPException(422, f"gallery: {error}")
        if model_version and model_version != emb.version:
            raise HTTPException(409, f"gallery embedded with model {model_version}, but {emb.version} is served")
        decisions = match_decisions(vectors, meds, emb)
        result["pills"] = [{"box": list(b), **d} for b, d in zip(box_list, decisions)]
        if schedule:
            try:
                sched = [ScheduledMedication(**s).model_dump() for s in parse_json(schedule, "schedule")]
            except (TypeError, ValidationError) as error:
                raise HTTPException(422, f"schedule: {error}")
            alerts, summary = check_dose(sched, [{"medication_id": d["medication_id"], "decision": d["decision"]}
                                                 for d in decisions],
                                         {m.medication_id: m.name or m.medication_id for m in meds})
            result["dose"] = {"alerts": [a.to_dict() for a in alerts], "summary": summary}
    else:
        result["pills"] = [{"box": list(b), **identify_pill(v[None], top_k=top_k)}
                           for b, v in zip(box_list, vectors)]
    return result


@app.get("/v1/pills/search")
def pills_search(name: str = Query(min_length=2, description="Brand or generic name, e.g. 'metformin', 'Glucophage'"),
                 strength: str | None = Query(None, description="e.g. '500 MG'"),
                 limit: int = Query(50, ge=1, le=200)):
    """
    **What does this medicine look like?** A name -> every catalogue product with it, with pictures. Products with
    exactly that medicine come first (`match: same`), then combinations containing it (`contains`), then name
    matches (`name`). One medicine usually has many different-looking pills (manufacturers, strengths).
    """
    cat = asset("catalogue")
    products, resolution = cat.search(name, app.state.rxnorm, strength, limit)
    return {"resolution": resolution, "count": len(products),
            "products": [product_summary(cat, p["pill_id"], match=p["match"]) for p in products],
            "source": cat.meta.get("source")}


@app.get("/v1/pills/{pill_id}")
def pill(pill_id: str):
    """One catalogue product: names, ingredients, appearance, imprint and its pictures."""
    cat = asset("catalogue")
    if cat.product(pill_id) is None:
        raise HTTPException(404, f"no catalogue pill {pill_id}")
    return product_summary(cat, pill_id)


@app.get("/v1/pills/{pill_id}/image/{n}")
def pill_image(pill_id: str, n: int):
    """A product's picture (JPEG): `rximage` (both sides, mm ruler), `reference` (studio) or `photo` (everyday)."""
    path = asset("catalogue").image_file(pill_id, n)
    if path is None or not path.exists():
        raise HTTPException(404, "no such picture")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


# ---- endpoints: feedback for retraining (opt-in) -----------------------------------------------------------------
@app.post("/v1/feedback")
async def feedback(image: UploadFile = File(description="The pill: a crop, or a photo with `box`"),
                   box: str | None = Form(None, description="Optional [x0, y0, x1, y1] of the pill in the photo"),
                   consent: bool = Form(description="The user agreed to share this photo to improve the model"),
                   consent_id: str = Form(description="The app's id of this user's consent record (letters, "
                                                      "digits, - and _); used to delete their photos on withdrawal"),
                   medication_name: str = Form(description="What the pill is, e.g. 'Metformin 500 mg'"),
                   label_source: Literal["pharmacist", "user_confirmed"] = Form(
                       description="pharmacist: checked by a pharmacist; user_confirmed: the user confirmed it"),
                   ndc: str | None = Form(None), rxcui: str | None = Form(None),
                   app_decision: str | None = Form(None, description="What the app had decided (match / unknown / "
                                                                     "ambiguous), for error analysis"),
                   app_medication_id: str | None = Form(None)):
    """
    **Share a verified pill photo for the monthly retraining** (opt-in; off unless the server is configured).
    Only the pill crop and its label are stored, never the full photo. Use it when a pharmacist has checked a pill,
    or the user confirmed what a pill is (e.g. after an "unknown" or a wrong match).
    """
    if not consent:
        raise HTTPException(422, "consent must be true: only photos the user agreed to share are stored")
    store = asset("feedback")
    photo = await read_image(image)
    b = parse_box(box)
    crop = crop_window(photo, b, getattr(app.state.embedder, "window", WINDOW)) if b else photo.resize((384, 384))
    meta = {"consent_id": consent_id, "medication_name": medication_name, "label_source": label_source,
            "ndc": ndc, "rxcui": rxcui, "app_decision": app_decision, "app_medication_id": app_medication_id,
            "model_version": getattr(app.state.embedder, "version", None), "api_version": API_VERSION,
            "received_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    try:
        item = store.save(crop, meta)
    except ValueError as error:
        raise HTTPException(422, str(error))
    return {"stored": True, "item": item, "note": "Thank you. The photo is used only to improve pill recognition."}


@app.delete("/v1/feedback/{consent_id}")
def feedback_delete(consent_id: str):
    """Withdrawal of consent: deletes every photo stored under this consent id."""
    try:
        return {"deleted": True, **asset("feedback").delete(consent_id)}
    except ValueError as error:
        raise HTTPException(422, str(error))


# ---- endpoints: Stage 3 -------------------------------------------------------------------------------------------
@app.post("/v1/safety/medications")
def safety_medications(request: MedicationsRequest):
    """
    Run when a medication is added or changed: interactions between every pair of the patient's medications
    (DDInter severity) and therapeutic duplications (the same active ingredient twice). Medications that can't be
    identified are reported as alerts, so an empty alert list means every medication was checked.
    """
    db = asset("safety_db")
    resolved = [resolve_medication(m.model_dump(), app.state.rxnorm) for m in request.medications]
    alerts, per_med = check_medications(resolved, db)
    return {
        "alerts": [a.to_dict() for a in alerts],
        "medications": per_med,
        "all_identified": all(m.resolved_by != "unresolved" for m in resolved),
        "sources": {"interactions": db.meta.get("source"), "licence": db.meta.get("licence"),
                    "built_at": db.meta.get("built_at")},
    }


@app.post("/v1/safety/dose")
def safety_dose(request: DoseRequest):
    """
    Run at every dose check, after /v1/match: double doses, pills not part of this dose, missed medications, and
    unknown or unsure pills. (/v1/scan with a schedule does this in the same call.)
    """
    alerts, summary = check_dose([s.model_dump() for s in request.schedule],
                                 [p.model_dump() for p in request.pills], request.names)
    return {"alerts": [a.to_dict() for a in alerts], "summary": summary}


@app.get("/v1/safety/label")
def safety_label(rxcui: str | None = Query(None), ingredient: str | None = Query(None)):
    """The warning sections of the official US label (openFDA), quoted as written; only when the user asks."""
    if not rxcui and not ingredient:
        raise HTTPException(422, "Give rxcui or ingredient")
    try:
        label = openfda.label_warnings(rxcui=rxcui, ingredient=ingredient)
    except Exception as error:
        raise HTTPException(502, f"openFDA lookup failed: {error}")
    if label is None:
        raise HTTPException(404, "No label found")
    return label
