"""
PharmaScan API: Stage 2 (pill verification) and Stage 3 (safety checks) for the app.

Stateless by design: no photo, gallery or schedule is stored on the server. The app keeps each patient's gallery
embeddings and schedule on the phone and sends what a request needs. The endpoints are described in
serving/README.md; only /health exists until the model is trained.

Run locally, from the epills folder:
    uvicorn serving.app.main:app --reload --port 7860
"""
import os

from fastapi import FastAPI

import epills.core  # noqa: F401  (the only project package the API may import)

API_VERSION = "0.1.0"

app = FastAPI(
    title="PharmaScan API",
    version=API_VERSION,
    description="Pill verification against a patient's enrolled medications, and medication safety checks. "
                "A medication verification aid, not a medical device: every alert defers to a pharmacist.",
)


@app.get("/health")
def health():
    """Liveness check, and which model the server would load."""
    return {
        "status": "ok",
        "api_version": API_VERSION,
        "model_repo": os.environ.get("EPILLS_MODEL_REPO", "not set"),
        "model_loaded": False,
    }
