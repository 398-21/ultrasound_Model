"""Smoke test of the hosted API (skipped unless serving/requirements.txt is installed)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # needed by FastAPI's TestClient
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "tests")]
from serving.app.main import app  # noqa: E402


def test_health():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_safety_endpoints(tmp_path):
    from epills.core.safety.db import SafetyDB, write_db
    from test_safety import DRUGS, INGREDIENTS, PAIRS, FakeRxNorm

    write_db(tmp_path / "safety.sqlite", DRUGS, INGREDIENTS, PAIRS, {"source": "test", "pairs": len(PAIRS)})
    saved = app.state.safety_db, app.state.rxnorm
    app.state.safety_db, app.state.rxnorm = SafetyDB(tmp_path / "safety.sqlite"), FakeRxNorm()
    client = TestClient(app)
    try:
        assert client.get("/health").json()["safety_db"]["loaded"] is True

        r = client.post("/v1/safety/medications", json={"medications": [
            {"id": "a", "name": "Lipitor 10 mg", "ndc": "00071-0155-23"}, {"id": "b", "name": "Clarithromycin"}]})
        assert r.status_code == 200
        body = r.json()
        assert body["all_identified"] is True
        assert [(a["type"], a["severity"], a["colour"]) for a in body["alerts"]] == [("interaction", "major", "red")]

        r = client.post("/v1/safety/dose", json={
            "schedule": [{"medication_id": "a", "count": 1}],
            "pills": [{"medication_id": "a", "decision": "match"}, {"decision": "unknown"}],
            "names": {"a": "Lipitor"}})
        assert [a["type"] for a in r.json()["alerts"]] == ["unknown_pill"]

        assert client.post("/v1/safety/medications", json={"medications": []}).status_code == 422
    finally:
        app.state.safety_db, app.state.rxnorm = saved
