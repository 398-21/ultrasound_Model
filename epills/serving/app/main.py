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
EPILLS_OCR=0 switches off text reading (imprints, boxes and labels); EPILLS_OCR_MODEL picks the Florence-2 model
(default florence-community/Florence-2-base, downloaded from Hugging Face on first use).
EPILLS_CORS_ORIGINS: comma-separated web origins allowed to call the API from a browser (default * , any).
EPILLS_API_KEY: if set, every /v1 request needs the header X-API-Key with this value.
EPILLS_FEEDBACK_REPO / EPILLS_FEEDBACK_DIR: opt-in photo collection for retraining (serving/app/feedback.py).

Run locally, from the epills folder:
    uvicorn serving.app.main:app --reload --port 7860
"""
import asyncio
import datetime
import io
import json
import os
import re
import tarfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import AfterValidator, BaseModel, Field, ValidationError
from pydantic_core import PydanticCustomError
from starlette.exceptions import HTTPException as StarletteHTTPException

from epills.core.catalogue.confidence import assess
from epills.core.catalogue.identity import product_identity
from epills.core.catalogue.store import IMPRINT_ALPHA, IMPRINT_TOP_K
from epills.core.imprint.match import GALLERY_CONFIRM_DISTINCT, GALLERY_MIN_WEIGHT, read_hits
from epills.core.imprint.ocr import GPU_MODEL
from epills.core.inference.matching import check_imprint, decide
from epills.core.inference.input_type import MESSAGES as INPUT_MESSAGES, classify_input
from epills.core.inference.pill_finder import detect_pills
from epills.core.label import STRENGTH, compare_with_list, document_fields
from epills.core.inference.preprocess import WINDOW, crop_window
from epills.core.safety import openfda
from epills.core.safety.db import SafetyDB
from epills.core.safety.rules import SOURCE_SCHEDULE, Alert, check_dose, check_medications, resolve_medication
from epills.core.safety.rxnorm import RxNormClient, RxNormError
from serving.app import evidence, schemas as S
from serving.app.vision import Vision

API_VERSION = "0.9.0"
STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
LOCAL = {"safety_db": Path("data/processed/safety/safety.sqlite"),
         "embedder": Path("outputs/serving/stage2_model.pt"),
         "catalogue": Path("data/processed/catalogue")}
IDENTIFY_NOTE = ("Possible matches from a catalogue of US pills, ranked by how alike they look. A photo can't confirm "
                 "what a pill is. Check with your pharmacist.")
STRENGTH_NOTE = ("Other strengths of this medicine look the same (look_alike_strengths): a photo can't tell the "
                 "strength. Check the box or label.")
LABEL_NOTE = ("Medicines read from the photo. Check the name and strength against the box or label before adding "
              "it to your list.")
VISUAL_LIMITS = ("A visual comparison with reference photos only: it can't confirm what the medicine is, its strength, "
                 "or that it is safe to take.")
NO_PILL = "No pill found. Spread the pills apart on a plain background and photograph them again."
SAFETY_NOTE = ("No alert doesn't mean these medicines are safe together: only interactions listed in DDInter 2.0 "
               "between identified ingredients, and the same ingredient twice, are checked. Check with your pharmacist.")
IMAGE_LICENCE = ("NLM C3PI pill images: for research and development use. A catalogue picture may not show the product "
                 "the patient holds.")
STAND_IN = "stand-in finder (plain backgrounds only)"
MATCH_MESSAGES = {  # visual evidence only (blueprint D5): "looks like", never "is"
    "match": "This pill looks like your {name}.",
    "unknown": "This pill doesn't look like any of your medicines. Set it aside. Check with your pharmacist.",
    "ambiguous": "This pill looks like more than one of your medicines. Turn it over and photograph it again.",
}
ERROR_CODES = {401: "unauthorised", 404: "not_found", 409: "version_mismatch", 422: "invalid_input",
               500: "model_error", 502: "upstream_failed", 503: "not_ready"}
TEXT_SKIPPED = ("The imprint wasn't read: the GPU is busy and reading on the CPU would take too long. Candidates are "
                "ranked by appearance only; type the imprint if you can see it.")
InputType = Literal["loose_pills", "blister", "package", "document"]
TEXT_UNAVAILABLE = ("The imprint couldn't be read (text reading is unavailable right now): candidates are ranked by "
                    "appearance only. Type the imprint if you can see it.")
RETRY_AFTER = 20  # seconds, with 503 not_ready (a part still loading, or the Space waking up)


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


def imprint_adapter_path():
    """The GPU reader's imprint adapter (scripts/train_imprint_reader.py --export), or None: then imprints are read by
    the original model. EPILLS_IMPRINT_ADAPTER=0 switches it off."""
    if os.environ.get("EPILLS_IMPRINT_ADAPTER") == "0":
        return None
    try:
        path = Path(_asset("EPILLS_IMPRINT_ADAPTER", "imprint_adapter.pt", Path("outputs/serving/imprint_adapter.pt")))
    except Exception as error:  # not uploaded: the original model reads
        print(f"no imprint adapter: {error}", flush=True)
        return None
    return path if path.exists() else None


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
    if name == "reader":
        if os.environ.get("EPILLS_OCR") == "0":
            raise RuntimeError("text reading is switched off (EPILLS_OCR=0)")
        from epills.core.imprint.ocr import CPU_MODEL, ImprintReader
        return ImprintReader(os.environ.get("EPILLS_OCR_MODEL", CPU_MODEL), device=os.environ.get("EPILLS_DEVICE", "cpu"),
                             quantize=os.environ.get("EPILLS_OCR_INT8", "1") == "1")
    if name == "label_reader":
        from epills.core.label import LabelReader
        return LabelReader(app.state.rxnorm)
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
            for name in ("safety_db", "embedder", "catalogue", "reader"):
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
                "ingredients: RxNorm (NLM); labels: openFDA; pill pictures: NLM C3PI (research and development use). "
                "Stage 2 answers are visual evidence only: they never confirm a medicine or that it is safe to take.",
    responses={code: {"model": S.Error, "description": name} for code, name in ERROR_CODES.items()},
)
app.state.rxnorm = RxNormClient()
app.state.safety_db = app.state.embedder = app.state.catalogue = app.state.feedback = None
app.state.reader = app.state.label_reader = app.state.vision = None
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("EPILLS_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def api_key(request: Request, call_next):
    """Optional shared key (EPILLS_API_KEY) on /v1 routes; /health and /docs stay open."""
    key = os.environ.get("EPILLS_API_KEY")
    if key and request.url.path.startswith("/v1") and request.method != "OPTIONS" \
            and request.headers.get("X-API-Key") != key:
        return JSONResponse({"code": "unauthorised", "detail": "missing or wrong X-API-Key"}, status_code=401)
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def http_error(_request, error):
    """One error shape everywhere (blueprint B5): code, detail, and retry_after with Retry-After on 503."""
    body, headers = {"code": ERROR_CODES.get(error.status_code, "error"), "detail": error.detail}, \
        dict(getattr(error, "headers", None) or {})
    if error.status_code == 503:
        body["retry_after"], headers["Retry-After"] = RETRY_AFTER, str(RETRY_AFTER)
    return JSONResponse(body, status_code=error.status_code, headers=headers)


@app.exception_handler(RequestValidationError)
async def invalid_input(_request, error):
    return JSONResponse({"code": "invalid_input", "detail": jsonable(error.errors())}, status_code=422)


def jsonable(errors):
    return [{k: (v if isinstance(v, (str, int, float, list, type(None))) else str(v)) for k, v in e.items()
             if not (k == "input" and e.get("type") == "personal_id")} for e in errors]


# Pseudonymous ids only (blueprint B26): anything shaped like a Singapore NRIC or FIN is refused and never echoed back
NRIC_LIKE = re.compile(r"(?<![A-Za-z0-9])[STFGM]\d{7}[A-Z](?![A-Za-z0-9])", re.IGNORECASE)


def no_personal_id(value):
    if value is not None and NRIC_LIKE.search(value):
        raise PydanticCustomError("personal_id", "looks like an NRIC or FIN number; send the app's pseudonymous id")
    return value


AppText = Annotated[str, AfterValidator(no_personal_id)]  # every id and name the app sends


# ---- request models ---------------------------------------------------------------------------------------------
class Medication(BaseModel):
    id: AppText = Field(description="The app's id for this medication")
    name: AppText | None = Field(None, description="Name as on the label, brand or generic, e.g. 'Lipitor 10 mg'")
    rxcui: str | None = Field(None, description="RxNorm code, if the app has it")
    ndc: str | None = Field(None, description="National Drug Code from a US bottle, e.g. '00071-0155-23'")
    ingredients: list[str] | None = Field(None, description="Active ingredients from the label; use this for "
                                                            "medicines without US codes")


class MedicationsRequest(BaseModel):
    medications: list[Medication] = Field(min_length=1)


class ScheduledMedication(BaseModel):
    medication_id: AppText
    count: int = Field(1, ge=0, description="Pills of this medication expected in this dose")


class CheckedPill(BaseModel):
    medication_id: AppText | None = Field(None, description="Best match from /v1/match (None if unknown)")
    decision: Literal["match", "unknown", "ambiguous"]


class DoseRequest(BaseModel):
    schedule: list[ScheduledMedication]
    pills: list[CheckedPill]
    names: dict[AppText, AppText] = Field(default_factory=dict, description="medication_id -> name, for messages")


class GalleryMedication(BaseModel):
    medication_id: AppText = Field(description="The app's pseudonymous id for the patient's medication (e.g. its "
                                             "patient_medication_id); never an NRIC, name or phone number")
    name: AppText | None = Field(None, description="Optional: shown in the answer, e.g. 'Metformin 500 mg'; without it "
                                                 "answers use medication_id")
    reference_set_id: AppText | None = Field(None, description="The app's id for the set of reference photos these "
                                                             "embeddings come from; returned with the result")
    embeddings: list[list[float]] = Field(min_length=1, description="From /v1/embed, one per enrolment photo")
    imprint: str | None = Field(None, description="The pill's imprint, e.g. 'TEVA;5729' (from /v1/pills/search): "
                                                  "lets the imprint read on a pill check the answer")


class MatchRequest(BaseModel):
    queries: list[list[float]] = Field(min_length=1, description="Embeddings of the pills to check (/v1/embed)")
    gallery: list[GalleryMedication] = Field(min_length=1, description="The patient's enrolled medications")
    model_version: str | None = Field(None, description="The model_version stored with the gallery's embeddings. "
                                                        "This or embedding_space_id is required")
    embedding_space_id: str | None = Field(None, description="The embedding_space_id stored with the gallery")
    query_reads: list[list[str]] | None = Field(None, description="Per query, its `imprint_reads` from /v1/embed "
                                                                  "(read_imprint=true): turns on the imprint check")


class EmbeddingSearchRequest(BaseModel):
    embeddings: list[list[float]] = Field(min_length=1, max_length=2, description="One pill: 1 or 2 embeddings "
                                                                                  "(front and back) from /v1/embed")
    model_version: str | None = Field(None, description="The model_version /v1/embed returned. This or "
                                                        "embedding_space_id is required")
    embedding_space_id: str | None = None
    imprint_reads: list[str] | None = Field(None, description="The `imprint_reads` /v1/embed returned "
                                                              "(read_imprint=true), of both photos together")
    imprint: str | None = Field(None, description="Letters or numbers on the pill, typed by the user")
    shape: str | None = None
    colour: str | None = None
    top_k: int = Field(5, ge=1, le=20)


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


def vision():
    """
    The request pipeline (serving/app/vision.py): at most one GPU call per request, readings reported as used,
    skipped or unavailable, never dropped silently (blueprint B21). Rebuilt when the model or reader changes.
    """
    emb = asset("embedder")
    try:
        reader = asset("reader")
    except HTTPException:
        reader = None
    v = app.state.vision
    if v is None or v.embedder is not emb or v.reader is not reader:
        v = app.state.vision = Vision(emb, reader)
    return v


def text_note(status):
    return {"unavailable": TEXT_UNAVAILABLE, "skipped": TEXT_SKIPPED}.get(status)


def input_check(declared, check, result=None):
    """
    Is this a photo of loose pills (blueprint B23)? The app's declared type comes first; otherwise the photo's
    caption is checked for strong words (core/inference/input_type.py). Anything but loose pills abstains.
    """
    if declared and declared != "loose_pills":
        return {"type": declared, "caption": None, "checked": False, "abstained": True,
                "message": INPUT_MESSAGES.get(declared, INPUT_MESSAGES["document"])}
    if not check:
        return {"type": declared or "unchecked", "caption": None, "checked": False, "abstained": False, "message": None}
    if result is None or result["status"]["caption"] != "used":
        return {"type": "unchecked", "caption": None, "checked": False, "abstained": False,
                "message": INPUT_MESSAGES["unchecked"]}
    kind = classify_input(result["caption"])
    return {"type": kind, "caption": result["caption"], "checked": True, "abstained": kind != "loose_pills",
            "message": None if kind == "loose_pills" else INPUT_MESSAGES[kind]}


def checked(vectors):
    """Embeddings as returned by the pipeline, checked (blueprint B22): finite unit vectors, else a model_error."""
    vectors = np.asarray(vectors, np.float32)
    norms = np.linalg.norm(vectors, axis=1) if vectors.size else np.ones(0)
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or np.abs(norms - 1).max(initial=0) > 1e-3:
        raise HTTPException(500, "the model returned invalid numbers for this photo, so nothing was compared. "
                                 "Try another photo.")
    return vectors


def token_weights():
    """Imprint token rarity over the catalogue (as validated), or None (every token counts as rare)."""
    try:
        return asset("catalogue").imprint_index.weight
    except HTTPException:
        return None


def space(emb):
    return getattr(emb, "embedding_space_id", emb.version)


def check_gallery_version(emb, model_version, embedding_space_id):
    """Embeddings compare only within one space (blueprint B3): the gallery must say which space it was made in."""
    if not model_version and not embedding_space_id:
        raise HTTPException(422, "send the model_version (or embedding_space_id) stored with the embeddings")
    if (model_version and model_version != emb.version) or (embedding_space_id and embedding_space_id != space(emb)):
        raise HTTPException(409, f"embeddings made with {embedding_space_id or model_version}, but {space(emb)} is "
                                 f"served: re-embed the photos with /v1/embed")


def detection_out(det=None, boxes=None):
    """The detection block: from the stand-in finder (det), or the boxes the app sent (trusted as given)."""
    if det is None:
        return {"boxes": [list(b) for b in boxes], "box_flags": [[] for _ in boxes], "pill_count": len(boxes),
                "count_reliable": bool(boxes), "flags": [] if boxes else ["no_pills"], "found_by": "boxes sent"}
    return {"boxes": [list(b) for b in det.boxes], "box_flags": det.box_flags, "pill_count": det.pill_count,
            "count_reliable": bool(det.boxes) and "count_uncertain" not in det.flags, "flags": det.flags,
            "found_by": STAND_IN}


def match_decisions(queries, gallery, embedder, reads=None, reading=None):
    """
    Per query: the decision, and with reads (per query, its OCR strings) the imprint check on it. reading is the
    text-reading status: "unavailable" makes imprint_check say so instead of null.
    """
    dim = queries.shape[1]
    for g in gallery:
        if any(len(e) != dim for e in g.embeddings):
            raise HTTPException(422, f"gallery embeddings for {g.medication_id} don't have {dim} numbers")
    names = {g.medication_id: g.name or g.medication_id for g in gallery}
    sets = {g.medication_id: g.reference_set_id for g in gallery}
    imprints = {g.medication_id: g.imprint for g in gallery if g.imprint}
    weight = token_weights() if reads is not None and imprints else None
    if reads is not None and imprints and weight is None:  # the check was validated with the catalogue's weights
        reads, reading = None, "unavailable"
    reading = reading if reading in ("unavailable", "skipped") else None
    cal = embedder.calibration
    out = []
    for i, d in enumerate(decide(queries, {g.medication_id: np.asarray(g.embeddings, np.float32) for g in gallery},
                                 cal["threshold"], cal["margin"])):
        status, read_for = (reading if reading and imprints else None), []
        if reads is not None:
            hits = read_hits(reads[i], imprints, weight, GALLERY_MIN_WEIGHT)
            distinct = (set(read_hits(reads[i], imprints, weight, GALLERY_MIN_WEIGHT, distinct=True))
                        if GALLERY_CONFIRM_DISTINCT else None)
            d, status = check_imprint(d, set(hits), set(imprints), cal["threshold"], distinct)
            read_for = [names[m] for m in hits]
        matched = d.decision == "match"
        out.append({
            "decision": d.decision,
            "medication_id": d.medication_id if matched else None,
            "name": names[d.medication_id] if matched else None,
            "reference_set_id": sets[d.medication_id] if matched else None,
            "similarity": round(d.similarity, 4),
            "closest": {"medication_id": d.medication_id, "name": names[d.medication_id],
                        "reference_set_id": sets[d.medication_id], "similarity": round(d.similarity, 4)},
            "runner_up": ({"medication_id": d.runner_up_id, "name": names[d.runner_up_id],
                           "reference_set_id": sets[d.runner_up_id], "similarity": round(d.runner_up_similarity, 4)}
                          if d.runner_up_id is not None else None),
            "message": MATCH_MESSAGES[d.decision].format(name=names[d.medication_id]),
            "imprint_check": status,
            "imprint_read_for": read_for,
        })
    return out


def product_summary(cat, pill_id, **extra):
    p = cat.product(pill_id)
    return {
        "pill_id": pill_id, "name": p["name"], "rxnorm_name": p["rxnorm_name"], "generic_name": p["generic_name"],
        "manufacturer": p["manufacturer"], "ingredients": [i["name"] for i in p["ingredients"]],
        # identity fields (blueprint B13): never merge products across strength, dose form or release
        **product_identity(p["rxnorm_name"]), "look_alike_strengths": cat.look_alike_strengths(pill_id),
        "ndc9": p["ndc9"], "ndc11": p["ndc11"], "rxcui": p["rxcui"],
        "rxnorm_status": p["rxnorm_status"], "market": "US",
        "shape": p["shape"], "colour": p["colour"], "size_mm": p["size_mm"], "imprint": p["imprint"],
        "images": [{"url": f"/v1/pills/{pill_id}/image/{i['n']}", "kind": i["kind"]} for i in p["images"]],
        "image_licence": IMAGE_LICENCE,
        **extra,
    }


def identify_pill(embeddings, imprint=None, shape=None, colour=None, top_k=5, reads=None):
    cat = catalogue_for_identify()
    candidates, notes = cat.identify(embeddings, top_k=max(top_k, 2), imprint=imprint, shape=shape, colour=colour,
                                     reads=reads)
    # calibrated on unfiltered searches only: with a typed imprint, shape or colour the confidence is left out
    assessment = None if (imprint or shape or colour) else assess(candidates, cat.model_version)
    if candidates and cat.look_alike_strengths(candidates[0]["pill_id"]):  # test case M02
        notes.append(STRENGTH_NOTE)
    return {"candidates": [product_summary(cat, c.pop("pill_id"), **c) for c in candidates[:top_k]], "notes": notes,
            "note": IDENTIFY_NOTE, **(assessment or {"confidence": None, "catalogue_fit": None, "verdict": None,
                                                     "verdict_message": None})}


# ---- endpoints: health ----------------------------------------------------------------------------------------------
@app.get("/health", response_model=S.Health)
def health():
    """
    Liveness check, and which parts are loaded. `ready` is true once the model, catalogue and safety database are
    all loaded. A sleeping Space answers with an HTML page (not JSON) while it wakes up: retry until this is JSON.
    """
    try:
        db = asset("safety_db")
        safety = {"loaded": True, "built_at": db.meta.get("built_at"), "pairs": db.meta.get("pairs")}
    except HTTPException as error:
        safety = {"loaded": False, "error": error.detail}
    emb, cat = app.state.embedder, app.state.catalogue
    # where readings run now: the Space's GpuOrCpu pipeline knows (zerogpu, or cpu after a refusal); else the CPU
    gpu = getattr(getattr(app.state.vision, "compute", None), "device", None)
    device = gpu or ("cpu" if emb is not None else None)
    return {
        "status": "ok",
        "ready": emb is not None and cat is not None and safety["loaded"],
        "api_version": API_VERSION,
        "started_at": STARTED_AT,
        "model_repo": os.environ.get("EPILLS_MODEL_REPO", "not set"),
        "model": {"loaded": emb is not None, "version": getattr(emb, "version", None),
                  "embedding_space_id": space(emb) if emb is not None else None,
                  "device": str(device) if device else None, "error": _errors.get("embedder")},
        "catalogue": {"loaded": cat is not None, "products": len(cat) if cat else None,
                      "model_version": getattr(cat, "model_version", None), "error": _errors.get("catalogue")},
        "safety_db": safety,
        "text_reading": {"loaded": app.state.reader is not None, "model": getattr(app.state.reader, "name", None),
                         "gpu_model": GPU_MODEL if gpu else None,
                         "gpu_imprint_adapter": getattr(app.state, "gpu_imprint_adapter", None) if gpu else None,
                         "error": _errors.get("reader")},
    }


@app.get("/v1/tools")
def tools():
    """
    Typed tool definitions for the Medication Care Agent (blueprint B14), generated from this API's OpenAPI document:
    name, description for the agent, input and output schema, and how to call each. See docs/agent_tools.md.
    """
    from serving.app.tools import build_tools
    return build_tools(app.openapi())


@app.get("/v1/model-info", response_model=S.ModelInfo)
def model_info():
    """
    The served model (blueprint B3, B25): service version vs model version, the embedding space and preprocessing that
    stored embeddings must match, the calibrated decision rule, what the model was trained on and where it is weak,
    how similarity is computed, and the measured error rates with the evaluation file each comes from.
    """
    emb = asset("embedder")
    cal = emb.calibration
    return {
        "api_version": API_VERSION, "model_version": emb.version, "embedding_space_id": space(emb),
        "preprocess_version": getattr(emb, "preprocess_version", None) or "unknown", "embedding_dim": 256,
        "normalized": True, "crop_window": float(getattr(emb, "window", WINDOW)),
        "threshold": cal["threshold"], "margin": cal["margin"],
        "calibration": {"rule": cal.get("rule"), "calibrated_on": cal.get("calibrated_on")},
        "imprint": {"gallery_min_weight": GALLERY_MIN_WEIGHT, "identify_alpha": IMPRINT_ALPHA,
                    "identify_top_k": IMPRINT_TOP_K},
        "training_coverage": "Loose-pill photos of US prescription pills (NLM C3PI): 2,362 medicines; catalogue of "
                             "4,936 US pill products. Not trained on Singapore products, packaging or blisters.",
        "known_limits": [
            "Other phones and lighting: 48% top-1 over 32 Hong Kong products (MEDISEG), against 83% on held-out C3PI.",
            "Similarities are not probabilities: pills that aren't in the catalogue still score about 0.93 median.",
            "Imprints are readable on about 47% of photos (the GPU reader; fewer when reading runs on the CPU).",
            "The stand-in pill finder works on plain backgrounds only; check count_reliable before counting pills.",
            "Visual evidence only: never confirms the medicine, its strength, or that it is safe to take.",
        ],
        "similarity": evidence.SIMILARITY,
        "error_rates": evidence.error_rates(emb.version),
        "evidence": "visual",
    }


# ---- endpoints: Stage 2 -------------------------------------------------------------------------------------------
@app.post("/v1/embed", response_model=S.Embedding)
async def embed(image: UploadFile = File(description="A pill crop, or a photo with `box`"),
                box: str | None = Form(None, description="Optional [x0, y0, x1, y1] of the pill in the photo; the "
                                                         "server then crops it with the crop contract"),
                read_imprint: bool = Form(False, description="Also read the pill's imprint (for /v1/match's "
                                                             "query_reads); about 0.3 s on GPU, seconds on CPU")):
    """
    A pill photo -> its 256-number embedding. Used at **enrolment** (the app stores the embeddings as the patient's
    gallery, with the model version) and before `/v1/match`. Without a box, the image must already be framed like a
    crop: the pill in the middle, its long side about 70% of the image.
    """
    emb = asset("embedder")
    photo = await read_image(image)
    b = parse_box(box)
    crops = emb.frame(photo, [b] if b else None)
    r = await run_in_threadpool(vision().run, crops, crops if read_imprint else None)
    vector = checked(r["embeddings"])[0]
    reads, reading = r["reads"], r["status"]["read"]
    return {"embedding": [round(float(v), 6) for v in vector], "dim": int(vector.shape[0]),
            "model_version": emb.version, "embedding_space_id": space(emb),
            "preprocess_version": getattr(emb, "preprocess_version", None) or "unknown", "normalized": True,
            "framed_by": "box" if b else "as sent", "imprint_reads": reads[0] if reads else None,
            "text_reading": reading, "evidence": "visual"}


@app.post("/v1/match", response_model=S.MatchResponse)
def match(request: MatchRequest):
    """
    **Which of my medicines is this?** Pill embeddings + the patient's gallery -> per pill: `match` (with the
    medication's name), `unknown` (not one of theirs: set it aside) or `ambiguous` (turn it over, photograph again).
    The threshold and margin are the served model's calibration.
    """
    emb = asset("embedder")
    check_gallery_version(emb, request.model_version, request.embedding_space_id)
    queries = np.asarray(request.queries, np.float32)
    if request.query_reads is not None and len(request.query_reads) != len(queries):
        raise HTTPException(422, "query_reads needs one list of reads per query")
    return {"results": match_decisions(queries, request.gallery, emb, request.query_reads),
            "model_version": emb.version, "embedding_space_id": space(emb),
            "threshold": emb.calibration["threshold"], "margin": emb.calibration["margin"],
            "evidence": "visual", "limits": VISUAL_LIMITS}


@app.post("/v1/search-by-embedding", response_model=S.IdentifyResponse)
def search_by_embedding(request: EmbeddingSearchRequest):
    """
    **What is this pill?**, from embeddings the app already has (blueprint B17): the same answer as `/v1/identify`
    for the same photos, without running the model again (e.g. after `/v1/match` said `unknown`). Send the
    `imprint_reads` from `/v1/embed` (read_imprint=true) to re-rank by the imprint. The photo isn't checked here:
    `/v1/embed` takes loose-pill photos only.
    """
    emb = asset("embedder")
    check_gallery_version(emb, request.model_version, request.embedding_space_id)
    vectors = np.asarray(request.embeddings, np.float32)
    dim = asset("catalogue").emb.shape[1]
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.shape[1] != dim or not np.isfinite(vectors).all() or np.abs(norms - 1).max() > 1e-3:
        raise HTTPException(422, f"embeddings must be the unit vectors of {dim} numbers that /v1/embed returned")
    reads = None if request.imprint else (request.imprint_reads or None)
    out = identify_pill(vectors, request.imprint, request.shape, request.colour, request.top_k, reads)
    return {**out, "imprint_reads": reads, "text_reading": "used" if reads else "not_requested", "input_check": None,
            "model_version": emb.version, "evidence": "visual", "limits": VISUAL_LIMITS}


@app.post("/v1/identify", response_model=S.IdentifyResponse)
async def identify(images: list[UploadFile] = File(description="One pill: 1 photo, or 2 (front and back)"),
                   boxes: str | None = Form(None, description="Optional [[x0, y0, x1, y1], ...], one per image"),
                   imprint: str | None = Form(None, description="Letters or numbers on the pill, e.g. 'M 547'"),
                   shape: str | None = Form(None, description="e.g. ROUND, OVAL, CAPSULE"),
                   colour: str | None = Form(None, description="e.g. WHITE, PINK"),
                   top_k: int = Form(5, ge=1, le=20),
                   read_imprint: bool = Form(True, description="Read the imprint from the photos and use it to "
                                                               "re-rank (skipped when `imprint` is typed)"),
                   input_type: InputType | None = Form(None, description="What the user says the photo shows; "
                                                                          "anything but loose_pills abstains"),
                   check_input: bool = Form(True, description="Check the photo shows loose pills (not a blister, "
                                                              "bottle or label); abstains otherwise")):
    """
    **What is this pill?** Photo(s) of one pill -> the most alike pills in the catalogue, with names and pictures.
    Typing the imprint narrows the list a lot. Always possible matches, never a confirmed identity.
    """
    if not 1 <= len(images) <= 2:
        raise HTTPException(422, "send 1 or 2 photos of the same pill")
    emb = asset("embedder")
    photos = [await read_image(i) for i in images]
    box_list = parse_boxes(boxes, len(photos))
    crops = [c for i, p in enumerate(photos) for c in emb.frame(p, [box_list[i]] if box_list else None)]
    base = {"model_version": emb.version, "evidence": "visual", "limits": VISUAL_LIMITS}
    early = input_check(input_type, check_input)
    if early["abstained"]:
        return {"candidates": [], "notes": [early["message"]], "note": IDENTIFY_NOTE, "input_check": early,
                "text_reading": "not_requested", **base}
    want_read = read_imprint and not imprint
    r = await run_in_threadpool(lambda: vision().run(crops, crops if want_read else None,
                                                     photos[0] if check_input and not input_type else None))
    ic = input_check(input_type, check_input and not input_type, r)
    if ic["abstained"]:
        return {"candidates": [], "notes": [ic["message"]], "note": IDENTIFY_NOTE, "input_check": ic,
                "text_reading": r["status"]["read"], **base}
    vectors = checked(r["embeddings"])
    reads = [x for crop_reads in r["reads"] for x in crop_reads] if r["reads"] else None
    reading = r["status"]["read"]
    out = identify_pill(vectors, imprint, shape, colour, top_k, reads)
    if text_note(reading):
        out["notes"].append(text_note(reading))
    return {**out, "imprint_reads": reads, "text_reading": reading, "input_check": ic, **base}


@app.post("/v1/detect", response_model=S.DetectResponse)
async def detect(image: UploadFile = File(description="A full photo of the pills"),
                 input_type: InputType | None = Form(None, description="What the user says the photo shows; "
                                                                        "anything but loose_pills abstains"),
                 check_input: bool = Form(True, description="Check the photo shows loose pills; abstains otherwise")):
    """
    **Where are the pills?** A photo -> a box per pill with quality flags (blueprint B6). Uses the stand-in finder for
    plain backgrounds until the Stage 1 detector replaces it (same box format). Count pills only when
    `count_reliable` is true; otherwise ask for a new photo (pills apart, plain surface).
    """
    photo = await read_image(image)
    ic = input_check(input_type, check_input)
    if not ic["abstained"] and check_input and not input_type:
        det, r = await asyncio.gather(run_in_threadpool(detect_pills, photo),
                                      run_in_threadpool(lambda: vision().run(caption=photo)))
        ic = input_check(None, True, r)
    elif not ic["abstained"]:
        det = await run_in_threadpool(detect_pills, photo)
    if ic["abstained"]:
        return {**detection_out(boxes=[]), "flags": ["no_pills", "not_loose_pills"], "found_by": STAND_IN,
                "input_check": ic, "message": ic["message"]}
    out = {**detection_out(det), "input_check": ic}
    if not out["boxes"]:
        out["message"] = NO_PILL
    return out


ROUTES = {  # input type -> the tools that fit it, and what to tell the user (blueprint flows ds_route, un_route)
    "loose_pills": (["detect_pills", "embed_pill", "match_gallery", "identify_pill"],
                    "Loose pills: compare them with the patient's medicines, or look them up in the catalogue."),
    "printed": (["extract_medication_document"], "Printed text: read the label."),
    "package": (["extract_medication_document"], "A box or bottle: read its label."),
    "document": (["extract_medication_document"], "A document or label: read it."),
    "blister": ([], "A blister pack: pills in it can't be recognised. Photograph the box or pharmacy label, or ask "
                    "the pharmacist."),
    "unchecked": ([], "The photo couldn't be checked automatically: ask the user what it shows."),
}


@app.post("/v1/check-input", response_model=S.CheckInputResponse)
async def check_input_photo(image: UploadFile = File(description="The photo the user sent"),
                            input_type: InputType | None = Form(None, description="What the user says it shows; "
                                                                                  "taken as given")):
    """
    **What does this photo show?** (blueprint B19, tool `route_medication_image`): loose pills, a blister pack, a box or
    bottle, printed text or a document, and which tools fit it. The user's own word comes first; otherwise the photo's
    description is checked (a net, not a guarantee: ask the user when `type` is `unchecked`). PDFs are documents: send
    them to `/v1/read-label` directly.
    """
    photo = await read_image(image)
    if input_type:
        ic = input_check(input_type, False)
    else:
        r = await run_in_threadpool(lambda: vision().run(caption=photo))
        ic = input_check(None, True, r)
    tools, message = ROUTES[ic["type"]]
    return {"input_check": ic, "next_tools": tools, "message": message}


@app.post("/v1/scan", response_model=S.ScanResponse)
async def scan(image: UploadFile = File(description="A full photo of the pills"),
               boxes: str | None = Form(None, description="Pill boxes from a detector, [[x0, y0, x1, y1], ...]; "
                                                          "without them a stand-in finder looks for pills on a "
                                                          "plain background"),
               gallery: str | None = Form(None, description="JSON list of {medication_id, name, embeddings}: the "
                                                            "patient's gallery. Without it, each pill is "
                                                            "identified against the catalogue"),
               schedule: str | None = Form(None, description="Deprecated (the main backend owns dose rules): JSON "
                                                             "list of {medication_id, count}; adds an advisory dose "
                                                             "check"),
               model_version: str | None = Form(None, description="The model_version stored with the gallery "
                                                                  "(this or embedding_space_id is required with one)"),
               embedding_space_id: str | None = Form(None, description="The embedding_space_id stored with the "
                                                                       "gallery"),
               top_k: int = Form(3, ge=1, le=10),
               read_imprint: bool = Form(True, description="Read each pill's imprint: checks the gallery decisions "
                                                           "(medications with an `imprint`), or re-ranks catalogue "
                                                           "candidates"),
               input_type: InputType | None = Form(None, description="What the user says the photo shows; anything "
                                                                      "but loose_pills abstains"),
               check_input: bool = Form(True, description="Check the photo shows loose pills; abstains otherwise")):
    """
    **Candidates for every pill in a photo** (diagnostics; blueprint D3): full photo -> each pill's box with quality
    flags and its visual decision (with the patient's gallery) or catalogue candidates (without one). Not a dose
    verification: the main backend decides with the regimen. Boxes flagged by the finder are not counted.
    """
    emb = asset("embedder")
    photo = await read_image(image)
    early = input_check(input_type, check_input)
    if early["abstained"]:
        return {"found_by": STAND_IN, "model_version": emb.version, "pills": [], "input_check": early,
                "detection": {**detection_out(boxes=[]), "flags": ["no_pills", "not_loose_pills"]},
                "message": early["message"], "evidence": "visual", "limits": VISUAL_LIMITS}
    box_list = parse_boxes(boxes)
    if box_list is None:
        detection = detection_out(await run_in_threadpool(detect_pills, photo))
    else:
        detection = detection_out(boxes=box_list)
    box_list = [tuple(b) for b in detection["boxes"]]
    result = {"found_by": detection["found_by"], "model_version": emb.version, "detection": detection, "pills": [],
              "evidence": "visual", "limits": VISUAL_LIMITS}
    if not box_list:  # say what it looks like instead, if it isn't loose pills (a blister, a bottle)
        if check_input and not input_type:
            ic = input_check(None, True, await run_in_threadpool(lambda: vision().run(caption=photo)))
            result["input_check"] = ic
            result["message"] = ic["message"] if ic["abstained"] else NO_PILL
        else:
            result["message"] = NO_PILL
        return result
    crops = emb.frame(photo, box_list)
    meds = None
    if gallery:
        try:
            meds = [GalleryMedication(**g) for g in parse_json(gallery, "gallery")]
        except TypeError as error:
            raise HTTPException(422, f"gallery: {error}")
        except ValidationError as error:
            raise HTTPException(422, {"gallery": jsonable(error.errors(include_input=False, include_url=False))})
        check_gallery_version(emb, model_version, embedding_space_id)
    want_read = read_imprint and (meds is None or any(m.imprint for m in meds))
    r = await run_in_threadpool(lambda: vision().run(crops, crops if want_read else None,
                                                     photo if check_input and not input_type else None))
    ic = input_check(input_type, check_input and not input_type, r)
    result["input_check"] = ic
    if ic["abstained"]:
        result.update(pills=[], message=ic["message"])
        return result
    vectors = checked(r["embeddings"])
    reads, reading = r["reads"], r["status"]["read"]
    result["text_reading"] = reading
    if text_note(reading):
        result["message"] = text_note(reading)

    if meds is not None:
        decisions = match_decisions(vectors, meds, emb, reads, reading)
        result["pills"] = [{"box": list(b), "quality_flags": fl, "counted": not fl, **d,
                            "imprint_reads": reads[i] if reads else None}
                           for i, (b, fl, d) in enumerate(zip(box_list, detection["box_flags"], decisions))]
        if schedule:
            try:
                sched = [ScheduledMedication(**s).model_dump() for s in parse_json(schedule, "schedule")]
            except (TypeError, ValidationError) as error:
                raise HTTPException(422, f"schedule: {error}")
            result["dose"] = advisory_dose(sched, result["pills"], detection, meds)
    else:
        result["pills"] = [{"box": list(b), "quality_flags": fl, "counted": not fl,
                            **identify_pill(v[None], top_k=top_k, reads=reads[i] if reads else None),
                            "imprint_reads": reads[i] if reads else None}
                           for i, (b, fl, v) in enumerate(zip(box_list, detection["box_flags"], vectors))]
    return result


def advisory_dose(sched, pills, detection, meds):
    """/v1/scan's deprecated dose check: only when the pill count is reliable, and only counted pills."""
    note = ("Advisory and deprecated: the main backend checks doses against the regimen and dose history "
            "(docs/dose_rules.md).")
    if not detection["count_reliable"]:
        alert = Alert("count_uncertain", "moderate", "I can't count the pills in this photo reliably. Spread them "
                      "apart on a plain surface and photograph them again.", [], SOURCE_SCHEDULE,
                      details={"flags": detection["flags"]})
        summary = {"pills_seen": len(pills), "matched": 0, "unknown": 0, "unsure": 0, "all_ok": False,
                   "all_visually_matched": False, "count_reliable": False,
                   "text": "I can't count the pills in this photo reliably."}
        return {"alerts": [alert.to_dict()], "summary": summary, "deprecated": True, "note": note}
    alerts, summary = check_dose(sched, [{"medication_id": p["medication_id"], "decision": p["decision"]}
                                         for p in pills if p["counted"]],
                                 {m.medication_id: m.name or m.medication_id for m in meds})
    return {"alerts": [a.to_dict() for a in alerts], "summary": summary, "deprecated": True, "note": note}


@app.post("/v1/read-label", response_model=S.ReadLabelResponse)
async def read_label(image: UploadFile = File(description="A photo of a medicine box, a pharmacy label or a "
                                                          "prescription (close up, straight on), or a PDF"),
                     medications: str | None = Form(None, description="Optional JSON list of the patient's "
                                                                      "medications, [{id, name, rxcui?, strength?}]: "
                                                                      "each medicine read is compared with it")):
    """
    **Which medicine is on this box, label or prescription?** Reads the printed text and finds the medicine names in
    it (RxNorm's ingredient and brand names, international names such as paracetamol too), with strength,
    ingredients, RxNorm code, and what is printed about each (directions, quantity, dose form, dates), every value
    with the line it was read from and `null` when not printed (blueprint B7). With `medications`, each medicine is
    compared with the patient's list: same medicine, different strength, or not on the list (B9). Send a confirmed
    candidate to `/v1/safety/medications` as `{"id", "name", "rxcui"}`. Always ask the user to confirm.
    Nothing is stored; only the lines that name or describe a medicine are returned (labels show the patient's name).
    """
    data = await image.read()
    reader = asset("reader")
    if data[:5] == b"%PDF-" or (image.filename or "").lower().endswith(".pdf"):
        lines, pages = await run_in_threadpool(pdf_lines, data)
        source = "pdf_text" if lines is not None else "pdf_scan"
    else:
        try:
            pages = [ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")]
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(422, f"{image.filename or 'image'} is not a readable image or PDF: {error}")
        lines, source = None, "photo"
    ocr_model = reader.name
    if lines is None:  # one call for all pages
        r = await run_in_threadpool(lambda: vision().run(text=pages))
        if r["status"]["text"] != "used":
            raise HTTPException(503, "text reading is unavailable right now")
        lines = r["lines"]
        ocr_model = GPU_MODEL if r["device"] == "zerogpu" else reader.name
    meds = parse_json(medications, "medications") if medications else None
    try:
        label_reader = asset("label_reader")
        candidates = await run_in_threadpool(label_reader.read, lines)
        document_fields(lines, candidates)
        if meds is not None:
            listed = await run_in_threadpool(resolve_list, meds)
            for c in candidates:
                c["list_check"] = compare_with_list(c, listed)
    except RxNormError as error:
        raise HTTPException(502, f"RxNorm didn't answer: {error}")
    result = {"candidates": candidates, "note": LABEL_NOTE, "ocr_model": ocr_model, "source": source}
    if not candidates:
        result["message"] = ("No medicine name found. Photograph the label closer, straight on and in good light, "
                             "or type the name.")
    return result


def pdf_lines(data, max_pages=3):
    """(lines, None) from a PDF's text layer when it has one, else (None, pages rendered as images) to be read."""
    from pypdf import PdfReader
    try:
        pages = PdfReader(io.BytesIO(data)).pages[:max_pages]
        lines = [line.strip() for page in pages for line in (page.extract_text() or "").splitlines() if line.strip()]
    except Exception as error:
        raise HTTPException(422, f"not a readable PDF: {error}")
    if sum(len(line) for line in lines) >= 20:
        return lines, None
    import pypdfium2
    pdf = pypdfium2.PdfDocument(data)
    return None, [pdf[i].render(scale=2).to_pil() for i in range(min(len(pdf), max_pages))]


def resolve_list(meds):
    """The patient's list for the comparison: each medication's ingredients (RxNorm) and printed strength."""
    out = []
    for m in meds:
        if not isinstance(m, dict) or not m.get("id"):
            raise HTTPException(422, "medications: each needs an id (and a name or rxcui)")
        r = resolve_medication({"id": str(m["id"]), "name": m.get("name"), "rxcui": m.get("rxcui")}, app.state.rxnorm)
        strength = m.get("strength") or next((x.group() for x in STRENGTH.finditer(m.get("name") or "")), None)
        out.append({"id": str(m["id"]), "name": m.get("name") or r.display_name,
                    "ingredients": {i.rxcui for i in r.ingredients if i.rxcui}, "strength": strength})
    return out


@app.get("/v1/pills/search", response_model=S.SearchResponse)
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


@app.get("/v1/pills/{pill_id}", response_model=S.Product)
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
@app.post("/v1/feedback", response_model=S.FeedbackStored)
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


@app.delete("/v1/feedback/{consent_id}", response_model=S.FeedbackDeleted)
def feedback_delete(consent_id: str):
    """
    Withdrawal of consent: deletes every photo stored under this consent id, here and in the dataset repo, squashes
    the repo's history, and checks before saying `deleted` (blueprint B15). Safe to repeat.
    """
    try:
        result = asset("feedback").delete(consent_id)
        return {"deleted": result["verified"], **result}
    except ValueError as error:
        raise HTTPException(422, str(error))


# ---- endpoints: Stage 3 -------------------------------------------------------------------------------------------
@app.post("/v1/safety/medications", response_model=S.MedicationsResponse)
def safety_medications(request: MedicationsRequest):
    """
    Run when a medication is added or changed: interactions between every pair of the patient's medications
    (DDInter severity) and therapeutic duplications (the same active ingredient twice). Medications that can't be
    identified are reported as alerts. `coverage` says what could actually be screened: **no alert does not mean the
    medicines are safe together** (blueprint B10).
    """
    db = asset("safety_db")
    resolved = [resolve_medication(m.model_dump(), app.state.rxnorm) for m in request.medications]
    alerts, per_med = check_medications(resolved, db)
    return {
        "alerts": [a.to_dict() for a in alerts],
        "medications": per_med,
        "all_identified": all(m.resolved_by != "unresolved" for m in resolved),
        "coverage": coverage(per_med),
        "note": SAFETY_NOTE,
        "sources": {"interactions": db.meta.get("source"), "licence": db.meta.get("licence"),
                    "built_at": db.meta.get("built_at")},
    }


def coverage(per_med):
    """What the interaction check could screen: medications with at least one ingredient in DDInter, and pairs."""
    screened = {m["id"] for m in per_med
                if len(m["ingredients"]) > len(m["not_in_interaction_database"])}
    ids = [m["id"] for m in per_med]
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    checked = sum(a in screened and b in screened for a, b in pairs)
    return {"medications": len(per_med), "screened": len(screened),
            "not_screened": [m["name"] for m in per_med if m["id"] not in screened],
            "ingredients_not_in_interaction_database": sorted({n for m in per_med
                                                               for n in m["not_in_interaction_database"]}),
            "pairs": len(pairs), "pairs_checked": checked, "pairs_not_checkable": len(pairs) - checked}


@app.post("/v1/safety/dose", response_model=S.DoseResponse)
def safety_dose(request: DoseRequest):
    """
    The dose rules' **reference implementation** (blueprint D3: the main backend owns the decision, with the regimen
    and dose history; spec and test cases in docs/dose_rules.md). After /v1/match: extra pills, pills not part of
    this dose, scheduled medicines not seen, and unknown or unsure pills. Visual evidence only.
    """
    alerts, summary = check_dose([s.model_dump() for s in request.schedule],
                                 [p.model_dump() for p in request.pills], request.names)
    return {"alerts": [a.to_dict() for a in alerts], "summary": summary}


@app.get("/v1/safety/label", response_model=S.LabelResponse)
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
