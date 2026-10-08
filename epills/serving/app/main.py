"""
PharmaScan API: Stage 2 (pill verification) and Stage 3 (safety checks) for the app.

Stateless by design: no photo, gallery or schedule is stored on the server. The app keeps each patient's gallery
embeddings and schedule on the phone and sends what a request needs. The endpoints are described in
serving/README.md. Stage 3 endpoints work now; /v1/embed and /v1/match follow once the model is trained.

Assets (not in git): the safety database, and later the model weights, come from
  EPILLS_SAFETY_DB   a local path to safety.sqlite, or
  EPILLS_MODEL_REPO  a Hugging Face repo holding safety.sqlite (and the weights), downloaded at start-up;
                     HF_TOKEN is needed if the repo is private.
Without either, the local build output data/processed/safety/safety.sqlite is used when it exists.

Run locally, from the epills folder:
    uvicorn serving.app.main:app --reload --port 7860
"""
import os
import threading
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from epills.core.safety import openfda
from epills.core.safety.db import SafetyDB
from epills.core.safety.rules import check_dose, check_medications, resolve_medication
from epills.core.safety.rxnorm import RxNormClient

API_VERSION = "0.2.0"
LOCAL_SAFETY_DB = Path("data/processed/safety/safety.sqlite")

app = FastAPI(
    title="PharmaScan API",
    version=API_VERSION,
    description="Pill verification against a patient's enrolled medications, and medication safety checks. "
                "A medication verification aid, not a medical device: every alert defers to a pharmacist. "
                "Interaction data: DDInter 2.0 (CC BY-NC-SA 4.0); ingredients: RxNorm (NLM); labels: openFDA.",
)
app.state.rxnorm = RxNormClient()
app.state.safety_db = None
_db_lock = threading.Lock()
_db_error = None


def _find_safety_db():
    if os.environ.get("EPILLS_SAFETY_DB"):
        return Path(os.environ["EPILLS_SAFETY_DB"])
    if os.environ.get("EPILLS_MODEL_REPO"):
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(os.environ["EPILLS_MODEL_REPO"], "safety.sqlite",
                                    token=os.environ.get("HF_TOKEN")))
    return LOCAL_SAFETY_DB


def safety_db():
    """The interaction database, loaded once; 503 if it isn't available."""
    global _db_error
    if app.state.safety_db is None:
        with _db_lock:
            if app.state.safety_db is None:
                try:
                    app.state.safety_db = SafetyDB(_find_safety_db())
                    _db_error = None
                except Exception as error:  # report, don't crash the whole API
                    _db_error = str(error)
    if app.state.safety_db is None:
        raise HTTPException(503, f"Safety database not available: {_db_error}")
    return app.state.safety_db


# ---- request and response models ------------------------------------------------------------------------------
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


# ---- endpoints ------------------------------------------------------------------------------------------------
@app.get("/health")
def health():
    """Liveness check, and which assets are loaded."""
    try:
        db = safety_db()
        safety = {"loaded": True, "built_at": db.meta.get("built_at"), "pairs": db.meta.get("pairs")}
    except HTTPException as error:
        safety = {"loaded": False, "error": error.detail}
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "model_repo": os.environ.get("EPILLS_MODEL_REPO", "not set"),
        "model_loaded": False,
        "safety_db": safety,
    }


@app.post("/v1/safety/medications")
def safety_medications(request: MedicationsRequest):
    """
    Run when a medication is added or changed: interactions between every pair of the patient's medications
    (DDInter severity) and therapeutic duplications (the same active ingredient twice). Medications that can't be
    identified are reported as alerts, so an empty alert list means every medication was checked.
    """
    db = safety_db()
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
    unknown or unsure pills.
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
