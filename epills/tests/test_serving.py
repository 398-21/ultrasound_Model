"""The hosted API (skipped unless serving/requirements.txt is installed); Stage 2 with a fake model."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


class FakeEmbedder:
    """Stands in for the 200 MB model: the embedding is the crop's centre colour (8 numbers)."""
    version = "fake-1"
    calibration = {"threshold": 0.9, "margin": 0.05}

    def __call__(self, crops):
        out = []
        for c in crops:
            a = np.asarray(c.convert("RGB"), np.float32) / 255
            h, w = a.shape[:2]
            r, g, b = a[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3).mean(axis=0)
            v = np.array([r, g, b, 1 - r, 1 - g, 1 - b, 0.1, 0.1], np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)

    def embed_photo(self, image, boxes=None):
        from epills.core.inference.preprocess import crop_window
        return self([crop_window(image, b) for b in boxes] if boxes else [image])


RED, BLUE, GREEN = (210, 30, 30), (30, 60, 210), (30, 180, 60)


def png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def stage2(tmp_path):
    """The API with the fake model and a two-product catalogue built with it."""
    from epills.core.catalogue.store import Catalogue, write_catalogue
    from test_catalogue import FakeRxNorm, mat_photo

    fake = FakeEmbedder()
    red, blue = (fake([Image.new("RGB", (64, 64), c)])[0] for c in (RED, BLUE))
    folder = tmp_path / "catalogue"
    (folder / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8), RED).save(folder / "images" / "R_1_0.jpg")
    products = [{"pill_id": "R_1", "name": "RED PILL 5 MG", "generic_name": "REDAMINE", "shape": "ROUND",
                 "colour": "RED", "imprint": "R;5"},
                {"pill_id": "B_1", "name": "BLUE PILL 10 MG", "generic_name": "BLUEMYCIN", "shape": "OVAL",
                 "colour": "BLUE", "imprint": "B;10"}]
    write_catalogue(folder, products, [("R_1", "6809", "metformin")], [("R_1", 0, "rximage", "R_1_0.jpg")],
                    [red, blue], ["R_1", "B_1"], ["reference", "reference"], {"model_version": fake.version})
    saved = app.state.embedder, app.state.catalogue, app.state.rxnorm
    app.state.embedder, app.state.catalogue, app.state.rxnorm = fake, Catalogue(folder), FakeRxNorm()
    try:
        yield TestClient(app), mat_photo
    finally:
        app.state.embedder, app.state.catalogue, app.state.rxnorm = saved


def test_embed_and_match(stage2):
    client, _ = stage2
    emb = {}
    for name, colour in (("red", RED), ("blue", BLUE), ("green", GREEN)):
        r = client.post("/v1/embed", files={"image": ("p.png", png(Image.new("RGB", (64, 64), colour)))})
        assert r.status_code == 200 and r.json()["model_version"] == "fake-1" and r.json()["dim"] == 8
        emb[name] = r.json()["embedding"]
    gallery = [{"medication_id": "m1", "name": "Red pill 5 mg", "embeddings": [emb["red"]]},
               {"medication_id": "m2", "name": "Blue pill 10 mg", "embeddings": [emb["blue"]]}]
    r = client.post("/v1/match", json={"queries": [emb["red"], emb["green"]], "gallery": gallery})
    first, second = r.json()["results"]
    assert (first["decision"], first["name"], first["message"]) == ("match", "Red pill 5 mg", "This is your Red pill 5 mg.")
    assert (second["decision"], second["medication_id"]) == ("unknown", None)
    assert client.post("/v1/match", json={"queries": [emb["red"]], "gallery": gallery,
                                          "model_version": "old"}).status_code == 409


def test_embed_with_a_box(stage2):
    client, mat_photo = stage2
    photo = png(mat_photo([((300, 300), (80, 50), RED)]))
    r = client.post("/v1/embed", files={"image": ("p.png", photo)}, data={"box": "220,250,380,350"})
    assert r.json()["framed_by"] == "box"
    assert client.post("/v1/embed", files={"image": ("p.png", photo)}, data={"box": "1,2,3"}).status_code == 422


def test_scan_with_gallery_and_schedule(stage2):
    client, mat_photo = stage2
    fake = FakeEmbedder()
    gallery = [{"medication_id": "m1", "name": "Red pill", "embeddings": fake([Image.new("RGB", (9, 9), RED)]).tolist()},
               {"medication_id": "m2", "name": "Blue pill", "embeddings": fake([Image.new("RGB", (9, 9), BLUE)]).tolist()}]
    photo = png(mat_photo([((300, 300), (80, 50), RED), ((800, 500), (70, 70), BLUE)]))
    r = client.post("/v1/scan", files={"image": ("p.png", photo)},
                    data={"gallery": json.dumps(gallery), "schedule": json.dumps([
                        {"medication_id": "m1", "count": 1}, {"medication_id": "m2", "count": 1}])})
    body = r.json()
    assert body["found_by"].startswith("stand-in")
    assert sorted(p["name"] for p in body["pills"]) == ["Blue pill", "Red pill"]
    assert body["dose"]["summary"]["all_ok"] is True
    empty = client.post("/v1/scan", files={"image": ("p.png", png(mat_photo([])))})
    assert empty.json()["pills"] == [] and "No pill found" in empty.json()["message"]


def test_identify_search_and_pictures(stage2):
    client, _ = stage2
    r = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))])
    body = r.json()
    assert body["candidates"][0]["pill_id"] == "R_1" and "pharmacist" in body["note"]
    assert body["candidates"][0]["images"][0]["url"] == "/v1/pills/R_1/image/0"
    r = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))],
                    data={"imprint": "B 10"})
    assert r.json()["candidates"][0]["pill_id"] == "B_1"

    r = client.get("/v1/pills/search", params={"name": "Glucophage"})
    assert [p["pill_id"] for p in r.json()["products"]] == ["R_1"] and r.json()["products"][0]["match"] == "same"
    assert client.get("/v1/pills/R_1").json()["ingredients"] == ["metformin"]
    pic = client.get("/v1/pills/R_1/image/0")
    assert pic.status_code == 200 and pic.headers["content-type"] == "image/jpeg"
    assert client.get("/v1/pills/R_1/image/3").status_code == 404
    assert client.get("/v1/pills/nope").status_code == 404


def test_catalogue_of_another_model_version_is_refused(stage2):
    client, _ = stage2
    app.state.catalogue.model_version = "other"
    r = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))])
    assert r.status_code == 503 and "rebuild" in r.json()["detail"]


def test_feedback_is_opt_in_and_deletable(tmp_path, monkeypatch):
    monkeypatch.setenv("EPILLS_FEEDBACK_DIR", str(tmp_path / "fb"))
    monkeypatch.delenv("EPILLS_FEEDBACK_REPO", raising=False)
    saved = app.state.feedback
    app.state.feedback = None
    client = TestClient(app)
    form = {"consent": "true", "consent_id": "user-42", "medication_name": "Metformin 500 mg",
            "label_source": "pharmacist", "box": "[10, 10, 50, 50]"}
    try:
        photo = png(Image.new("RGB", (100, 100), RED))
        r = client.post("/v1/feedback", files={"image": ("p.png", photo)}, data=form)
        assert r.status_code == 200 and r.json()["stored"]
        item = r.json()["item"]
        meta = json.loads((tmp_path / "fb" / "user-42" / f"{item}.json").read_text())
        assert meta["medication_name"] == "Metformin 500 mg" and meta["label_source"] == "pharmacist"
        assert Image.open(tmp_path / "fb" / "user-42" / f"{item}.jpg").size == (384, 384)
        assert client.post("/v1/feedback", files={"image": ("p.png", photo)},
                           data={**form, "consent": "false"}).status_code == 422
        r = client.delete("/v1/feedback/user-42")
        assert r.json()["local_photos_removed"] == 1 and not (tmp_path / "fb" / "user-42").exists()
    finally:
        app.state.feedback = saved


def test_api_key_and_cors(monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv("EPILLS_API_KEY", "secret")
    assert client.get("/v1/pills/search", params={"name": "x"}).status_code == 401
    assert client.get("/health").status_code == 200  # stays open
    monkeypatch.delenv("EPILLS_API_KEY")
    r = client.options("/v1/match", headers={"Origin": "https://example.org", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") in ("*", "https://example.org")


def test_catalogue_archive_is_unpacked_once(tmp_path):
    import tarfile
    from serving.app.main import _unpacked
    src = tmp_path / "cat"
    src.mkdir()
    (src / "catalogue.sqlite").write_text("x")
    with tarfile.open(tmp_path / "catalogue.tar", "w") as tar:
        tar.add(src / "catalogue.sqlite", arcname="catalogue.sqlite")
    folder = _unpacked(tmp_path / "catalogue.tar")
    assert (folder / "catalogue.sqlite").read_text() == "x" and _unpacked(tmp_path / "catalogue.tar") == folder


def test_space_uses_the_gpu_and_falls_back_to_the_cpu():
    from serving.app.fallback import GpuOrCpu

    now, refuse, calls = [0.0], [False], []

    def gpu(x):
        calls.append("gpu")
        if refuse[0]:
            raise RuntimeError("You have exceeded your free ZeroGPU quota")
        return x * 2

    forward = GpuOrCpu(gpu, lambda x: calls.append("cpu") or x * 2, cooldown=600, clock=lambda: now[0])
    assert forward(1) == 2 and forward.device == "zerogpu"
    refuse[0] = True
    assert forward(1) == 2 and forward.device.startswith("cpu")      # refused: the CPU answers the same call
    refuse[0] = False
    now[0] = 599
    assert forward(1) == 2 and calls[-1] == "cpu"                       # cooling down: GPU not asked
    now[0] = 600
    assert forward(1) == 2 and calls[-1] == "gpu" and forward.device == "zerogpu"
    assert calls == ["gpu", "gpu", "cpu", "cpu", "gpu"]
