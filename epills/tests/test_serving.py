"""The hosted API (skipped unless serving/requirements.txt is installed); Stage 2 with a fake model."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
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
        assert body["coverage"]["pairs"] == 1 and "safe together" in body["note"]
        assert body["coverage"]["pairs"] == 1 and "safe together" in body["note"]
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

    def transform(self, crop):
        a = np.asarray(crop.convert("RGB"), np.float32) / 255
        h, w = a.shape[:2]
        return torch.from_numpy(a[h // 3:2 * h // 3, w // 3:2 * w // 3].reshape(-1, 3).mean(axis=0))

    def forward(self, x):
        r, g, b = (x[:, i].numpy() for i in range(3))
        v = np.stack([r, g, b, 1 - r, 1 - g, 1 - b, np.full_like(r, 0.1), np.full_like(r, 0.1)], axis=1)
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)

    def __call__(self, crops):
        return self.forward(torch.stack([self.transform(c) for c in crops]))

    def frame(self, image, boxes=None):
        from epills.core.inference.preprocess import crop_window
        return [crop_window(image, b) for b in boxes] if boxes else [image]

    def embed_photo(self, image, boxes=None):
        return self(self.frame(image, boxes))


RED, BLUE, GREEN = (210, 30, 30), (30, 60, 210), (30, 180, 60)


class FakeReader:
    """
    Stands in for Florence-2, with the reader's prepare / generate / finish interface: red pills read 'RX 55', blue
    ones 'BL10' (in one of four rotations); a dark photo is captioned as a blister pack, anything else as a pill.
    """
    name = "fake-ocr"

    def _job(self, task, images=1, **extra):
        return {"task": task, "ids": torch.zeros(images, 1), "px": torch.zeros(images, 1), "tokens": 1,
                "images": images, **extra}

    def prepare_read(self, crops, centre=0.92):
        return self._job("read", 4 * len(crops), crops=crops)

    def prepare_caption(self, image):
        return self._job("caption", image=image)

    def prepare_text(self, image):
        return self._job("text", image=image)

    def generate(self, ids, px, tokens, task="read"):
        return ids

    def finish(self, job, ids):
        if job["task"] == "read":
            return self.read(job["crops"])
        if job["task"] == "caption":
            dark = np.asarray(job["image"].convert("L"), np.float32).mean() < 40
            return "A close up of a blister pack of pills." if dark else "A white pill sitting on top of a table."
        return self.read_text(job["image"])

    def read(self, crops, centre=0.92):
        out = []
        for c in crops:
            r, g, b = np.asarray(c.convert("RGB"), np.float32)[c.height // 2, c.width // 2]
            text = "RX 55" if r > 150 and b < 100 else "BL10" if b > 150 else ""
            out.append([text, "", "", ""])
        return out

    def read_text(self, image):
        return ["EXAMPLE CLINIC", "GLUCOPHAGE 500MG TABLET", "Take 1 tablet twice a day after meals",
                "Qty: 60 tablets", "Name: MR A TAN"]


class FakeLabelReader:
    def read(self, lines):
        return [{"name": "Glucophage 500 mg", "brand": "Glucophage", "strength": "500 mg", "rxcui": "861007",
                 "rxnorm_name": None, "ingredients": [{"rxcui": "6809", "name": "metformin"}],
                 "found_in": [lines[1]], "first_line": 1}] if len(lines) > 1 else []


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
    names = ("embedder", "catalogue", "rxnorm", "reader", "label_reader")
    saved = [getattr(app.state, n) for n in names]
    for n, value in zip(names, (fake, Catalogue(folder), FakeRxNorm(), FakeReader(), FakeLabelReader())):
        setattr(app.state, n, value)
    try:
        yield TestClient(app), mat_photo
    finally:
        for n, value in zip(names, saved):
            setattr(app.state, n, value)


def test_embed_and_match(stage2):
    client, _ = stage2
    emb = {}
    for name, colour in (("red", RED), ("blue", BLUE), ("green", GREEN)):
        r = client.post("/v1/embed", files={"image": ("p.png", png(Image.new("RGB", (64, 64), colour)))})
        assert r.status_code == 200 and r.json()["model_version"] == "fake-1" and r.json()["dim"] == 8
        emb[name] = r.json()["embedding"]
    gallery = [{"medication_id": "m1", "name": "Red pill 5 mg", "embeddings": [emb["red"]]},
               {"medication_id": "m2", "name": "Blue pill 10 mg", "embeddings": [emb["blue"]]}]
    r = client.post("/v1/match", json={"queries": [emb["red"], emb["green"]], "gallery": gallery,
                                       "model_version": "fake-1"})
    first, second = r.json()["results"]
    assert (first["decision"], first["name"], first["message"]) == \
        ("match", "Red pill 5 mg", "This pill looks like your Red pill 5 mg.")
    assert r.json()["evidence"] == "visual" and "safe to take" in r.json()["limits"]
    missing = client.post("/v1/match", json={"queries": [emb["red"]], "gallery": gallery})
    assert missing.status_code == 422 and missing.json()["code"] == "invalid_input"  # the version is required
    assert (second["decision"], second["medication_id"]) == ("unknown", None)
    assert client.post("/v1/match", json={"queries": [emb["red"]], "gallery": gallery,
                                          "model_version": "old"}).status_code == 409
    assert client.post("/v1/match", json={"queries": [emb["red"]], "gallery": gallery,
                                          "embedding_space_id": "other-space"}).json()["code"] == "version_mismatch"


def test_embed_with_a_box(stage2):
    client, mat_photo = stage2
    photo = png(mat_photo([((300, 300), (80, 50), RED)]))
    r = client.post("/v1/embed", files={"image": ("p.png", photo)}, data={"box": "220,250,380,350"})
    assert r.json()["framed_by"] == "box"
    assert client.post("/v1/embed", files={"image": ("p.png", photo)}, data={"box": "1,2,3"}).status_code == 422


def test_match_takes_pseudonymous_ids_and_passes_reference_sets_through(stage2):
    client, mat_photo = stage2
    red = FakeEmbedder()([Image.new("RGB", (9, 9), RED)]).tolist()
    gallery = [{"medication_id": "pm-17", "reference_set_id": "rs-17-loose-v2", "embeddings": red}]
    body = {"queries": red, "gallery": gallery, "model_version": "fake-1"}
    r = client.post("/v1/match", json=body).json()["results"][0]
    assert (r["decision"], r["name"], r["reference_set_id"]) == ("match", "pm-17", "rs-17-loose-v2")  # name optional
    assert r["closest"]["reference_set_id"] == "rs-17-loose-v2"
    for field, value in (("medication_id", "S1234567D"), ("reference_set_id", "rs-t7654321j"),
                         ("name", "Metformin for G1234567X")):
        bad = client.post("/v1/match", json={**body, "gallery": [{**gallery[0], field: value}]})
        assert bad.status_code == 422 and bad.json()["detail"][0]["type"] == "personal_id"
        assert value not in bad.text  # never echoed back
    photo = png(mat_photo([((300, 300), (80, 50), RED)]))
    bad = client.post("/v1/scan", files={"image": ("p.png", photo)},
                      data={"gallery": json.dumps([{**gallery[0], "medication_id": "F7654321Q"}]),
                            "model_version": "fake-1"})
    assert bad.status_code == 422 and "personal_id" in bad.text and "F7654321Q" not in bad.text
    dose = client.post("/v1/safety/dose", json={"schedule": [{"medication_id": "pm-17", "count": 1}], "pills": [],
                                                 "names": {"pm-17": "Metformin (T0123456J)"}})
    assert dose.status_code == 422 and "T0123456J" not in dose.text
    for fine in ("S12345678", "pm-S1234567", "TS1234567AB"):  # not shaped like an NRIC on its own
        assert client.post("/v1/match", json={**body, "gallery": [{**gallery[0], "medication_id": fine}]}).status_code \
            == 200


def test_scan_with_gallery_and_schedule(stage2):
    client, mat_photo = stage2
    fake = FakeEmbedder()
    gallery = [{"medication_id": "m1", "name": "Red pill", "embeddings": fake([Image.new("RGB", (9, 9), RED)]).tolist()},
               {"medication_id": "m2", "name": "Blue pill", "embeddings": fake([Image.new("RGB", (9, 9), BLUE)]).tolist()}]
    photo = png(mat_photo([((300, 300), (80, 50), RED), ((800, 500), (70, 70), BLUE)]))
    r = client.post("/v1/scan", files={"image": ("p.png", photo)},
                    data={"gallery": json.dumps(gallery), "model_version": "fake-1", "schedule": json.dumps([
                        {"medication_id": "m1", "count": 1}, {"medication_id": "m2", "count": 1}])})
    body = r.json()
    assert body["found_by"].startswith("stand-in")
    assert sorted(p["name"] for p in body["pills"]) == ["Blue pill", "Red pill"]
    assert body["detection"]["count_reliable"] and body["dose"]["summary"]["all_visually_matched"] is True
    assert body["dose"]["deprecated"] is True
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


def test_search_by_embedding_gives_the_identify_answer(stage2):
    client, _ = stage2
    photo = png(Image.new("RGB", (64, 64), RED))
    e = client.post("/v1/embed", files={"image": ("p.png", photo)}).json()
    direct = client.post("/v1/identify", files=[("images", ("p.png", photo))], data={"read_imprint": "false"}).json()
    body = {"embeddings": [e["embedding"]], "model_version": e["model_version"]}
    again = client.post("/v1/search-by-embedding", json=body).json()
    assert [c["pill_id"] for c in again["candidates"]] == [c["pill_id"] for c in direct["candidates"]]
    assert again["confidence"] == direct["confidence"] and again["evidence"] == "visual"
    typed = client.post("/v1/search-by-embedding", json={**body, "imprint": "B 10"}).json()
    assert typed["candidates"][0]["pill_id"] == "B_1"
    assert client.post("/v1/search-by-embedding", json={**body, "model_version": "old"}).status_code == 409
    assert client.post("/v1/search-by-embedding", json={"embeddings": [e["embedding"]]}).status_code == 422
    bad = client.post("/v1/search-by-embedding", json={**body, "embeddings": [[0.5] * len(e["embedding"])]})
    assert bad.status_code == 422 and "unit vectors" in bad.json()["detail"]


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
        assert meta["review"] == "not_needed"
        own = client.post("/v1/feedback", files={"image": ("p.png", photo)},
                          data={**form, "label_source": "user_confirmed"}).json()["item"]
        assert json.loads((tmp_path / "fb" / "user-42" / f"{own}.json").read_text())["review"] == "pending"
        r = client.delete("/v1/feedback/user-42")
        assert r.json()["local_photos_removed"] == 2 and not (tmp_path / "fb" / "user-42").exists()
        assert r.json()["deleted"] and r.json()["verified"]
        audit = (tmp_path / "fb" / "audit.jsonl").read_text()
        assert [json.loads(x)["action"] for x in audit.splitlines()] == ["stored", "stored", "withdrawn"]
        assert "user-42" not in audit  # a hash of the consent id only
        assert client.delete("/v1/feedback/user-42").json()["deleted"]  # safe to repeat
    finally:
        app.state.feedback = saved


def test_withdrawal_squashes_the_repo_history_and_checks(tmp_path, monkeypatch):
    import huggingface_hub
    from serving.app.feedback import FeedbackStore
    calls = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def delete_folder(self, path_in_repo, **kw):
            calls.append(("delete", path_in_repo))

        def super_squash_history(self, **kw):
            calls.append(("squash", kw["repo_id"]))

        def list_repo_files(self, repo_id, **kw):
            return ["feedback/audit.jsonl", "feedback/other-7/a.jpg"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    store = FeedbackStore(tmp_path)
    store.repo_id = "me/pharmascan-feedback"
    out = store.delete("user-42")
    assert calls == [("delete", "feedback/user-42"), ("squash", "me/pharmascan-feedback")]
    assert out["removed_from_repo"] and out["history_squashed"] and out["verified"]


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


def test_scan_imprint_check_confirms_and_flags_a_conflict(stage2):
    client, mat_photo = stage2
    fake = FakeEmbedder()
    red, blue = (fake([Image.new("RGB", (9, 9), c)]).tolist() for c in (RED, BLUE))
    photo = png(mat_photo([((300, 300), (80, 50), RED), ((800, 500), (70, 70), BLUE)]))

    def scan(red_imprint, blue_imprint):
        gallery = [{"medication_id": "m1", "name": "Red pill", "embeddings": red, "imprint": red_imprint},
                   {"medication_id": "m2", "name": "Blue pill", "embeddings": blue, "imprint": blue_imprint}]
        pills = client.post("/v1/scan", files={"image": ("p.png", photo)},
                            data={"gallery": json.dumps(gallery), "model_version": "fake-1"}).json()["pills"]
        return {p["closest"]["name"]: (p["decision"], p["imprint_check"]) for p in pills}

    assert scan("RX;55", "BL;10") == {"Red pill": ("match", "confirmed"), "Blue pill": ("match", "confirmed")}
    # the imprints are swapped: each pill looks like one medicine but reads like the other
    assert scan("BL;10", "RX;55") == {"Red pill": ("ambiguous", "conflict"), "Blue pill": ("ambiguous", "conflict")}


def test_embed_reads_for_match_and_identify_rereads(stage2):
    client, _ = stage2
    r = client.post("/v1/embed", files={"image": ("p.png", png(Image.new("RGB", (64, 64), RED)))},
                    data={"read_imprint": "true"})
    assert r.json()["imprint_reads"] == ["RX 55", "", "", ""]
    query = r.json()["embedding"]
    gallery = [{"medication_id": "m1", "name": "Red pill", "embeddings": [query], "imprint": "RX;55"}]
    result = client.post("/v1/match", json={"queries": [query], "gallery": gallery, "model_version": "fake-1",
                                            "query_reads": [r.json()["imprint_reads"]]}).json()["results"][0]
    assert (result["decision"], result["imprint_check"], result["imprint_read_for"]) == \
        ("match", "confirmed", ["Red pill"])
    assert client.post("/v1/match", json={"queries": [query], "gallery": gallery, "model_version": "fake-1",
                                          "query_reads": []}).status_code == 422
    body = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))]).json()
    assert body["imprint_reads"] == ["RX 55", "", "", ""] and "imprint_read" in body["candidates"][0]
    assert body["verdict"] is None and body["confidence"] is None  # calibrated for the real model only
    typed = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))],
                        data={"imprint": "B 10"}).json()
    assert typed["imprint_reads"] is None  # a typed imprint is better than a read one


def test_read_label(stage2):
    client, _ = stage2
    body = client.post("/v1/read-label", files={"image": ("p.png", png(Image.new("RGB", (64, 64), "white")))}).json()
    assert body["candidates"][0]["name"] == "Glucophage 500 mg" and "Check the name" in body["note"]
    app.state.label_reader = FakeLabelReader()
    app.state.reader.read_text = lambda image: ["blank"]
    body = client.post("/v1/read-label", files={"image": ("p.png", png(Image.new("RGB", (64, 64), "white")))}).json()
    assert body["candidates"] == [] and "No medicine name found" in body["message"]


def test_model_info_detect_and_the_documented_contract(stage2):
    client, mat_photo = stage2
    info = client.get("/v1/model-info").json()
    assert info["model_version"] == "fake-1" and info["evidence"] == "visual" and info["known_limits"]
    assert "cosine" in info["similarity"]["measure"] and info["error_rates"] == []  # rates are run 4's, not fake-1's
    e = client.post("/v1/embed", files={"image": ("p.png", png(Image.new("RGB", (64, 64), RED)))}).json()
    assert e["embedding_space_id"] == "fake-1" and e["normalized"] is True and e["evidence"] == "visual"
    det = client.post("/v1/detect", files={"image": ("p.png", png(mat_photo([((300, 300), (80, 50), RED),
                                                                               ((800, 500), (70, 70), BLUE)])))}).json()
    assert det["pill_count"] == 2 and det["count_reliable"] and det["found_by"].startswith("stand-in")
    none = client.post("/v1/detect", files={"image": ("p.png", png(mat_photo([])))}).json()
    assert none["pill_count"] == 0 and not none["count_reliable"] and "No pill found" in none["message"]
    health = client.get("/health").json()
    assert "ready" in health and health["started_at"]
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    assert {"Health", "ModelInfo", "Embedding", "MatchResponse", "IdentifyResponse", "ScanResponse",
            "DetectResponse", "ReadLabelResponse", "Error"} <= set(schemas)


def test_model_info_error_rates_match_their_sources():
    from serving.app import evidence
    entries = evidence.error_rates(evidence.MODEL_VERSION)
    assert entries and all("scale" not in e for e in entries)
    checked = 0
    for e in evidence.ERROR_RATES:
        assert set(e["keys"]) <= set(e["rates"]) and all(0 <= r <= 1 for r in e["rates"].values())
        source = ROOT / e["source"]
        if not e["keys"] or not source.exists():  # evaluation outputs aren't in every checkout
            continue
        data = json.loads(source.read_text())
        for name, path in e["keys"].items():
            assert evidence.lookup(data, path) / e.get("scale", 1) == pytest.approx(e["rates"][name], abs=6e-4), name
            checked += 1
    if (ROOT / evidence.RUN4).exists():
        assert checked >= 20
    # the rates belong to the served model, named as it names itself (the catalogue is built for it)
    from epills.core.catalogue.confidence import MODEL_VERSION
    assert evidence.MODEL_VERSION == MODEL_VERSION
    catalogue = ROOT / "data/processed/catalogue"
    if (catalogue / "catalogue.sqlite").exists():
        from epills.core.catalogue.store import Catalogue
        assert Catalogue(catalogue).model_version == evidence.MODEL_VERSION


def test_not_ready_says_when_to_retry(stage2):
    client, _ = stage2
    app.state.catalogue.model_version = "other"
    r = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))])
    assert r.status_code == 503 and r.json()["code"] == "not_ready" and r.headers["Retry-After"] == "20"


def test_scan_never_counts_pills_when_the_finder_is_unsure(stage2, monkeypatch):
    import serving.app.main as main
    from epills.core.inference.pill_finder import Detection
    client, mat_photo = stage2
    fake = FakeEmbedder()
    gallery = [{"medication_id": "m1", "name": "Red pill", "embeddings": fake([Image.new("RGB", (9, 9), RED)]).tolist()}]
    unsure = Detection(boxes=[(260.0, 275.0, 340.0, 325.0), (100.0, 100.0, 120.0, 118.0)],
                       box_flags=[[], ["small"]], flags=["count_uncertain"])
    monkeypatch.setattr(main, "detect_pills", lambda photo: unsure)
    body = client.post("/v1/scan", files={"image": ("p.png", png(mat_photo([((300, 300), (80, 50), RED)])))},
                       data={"gallery": json.dumps(gallery), "model_version": "fake-1",
                             "schedule": json.dumps([{"medication_id": "m1", "count": 1}])}).json()
    assert [p["counted"] for p in body["pills"]] == [True, False]
    assert [a["type"] for a in body["dose"]["alerts"]] == ["count_uncertain"]
    assert body["dose"]["summary"]["count_reliable"] is False


def text_pdf(lines):
    """A one-page PDF with a text layer (no PDF library needed)."""
    text = " ".join(f"({t}) Tj 0 -16 Td" for t in lines)
    stream = f"BT /F1 12 Tf 72 720 Td {text} ET".encode()
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
               b"/Resources << /Font << /F1 5 0 R >> >> >>",
               b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_read_label_gives_printed_fields_and_compares_with_the_list(stage2):
    client, _ = stage2
    photo = png(Image.new("RGB", (64, 64), "white"))
    meds = json.dumps([{"id": "m1", "name": "Metformin", "strength": "850 mg"}, {"id": "m2", "name": "Lipitor"}])
    body = client.post("/v1/read-label", files={"image": ("p.png", photo)}, data={"medications": meds}).json()
    c = body["candidates"][0]
    assert body["source"] == "photo"
    assert c["fields"]["directions"]["value"] == "Take 1 tablet twice a day after meals"
    assert c["fields"]["quantity"]["value"] == "60 tablets" and c["fields"]["expiry_date"] is None
    assert c["list_check"]["status"] == "different_strength" and c["list_check"]["medication_id"] == "m1"
    assert "MR A TAN" not in json.dumps(body)  # the patient's name never comes back
    i = c["fields"]["instructions"]  # the sticker's parts, structured
    assert body["label_kind"] == "dispensing" and c["fields"]["quantity"]["amount"] == 60
    assert (i["dose"]["amount"], i["dose"]["unit"], i["times_per_day"], i["food"]) == (1, "tablet", 2, "after_food")
    assert c["fields"]["cautions"] == [] and c["fields"]["indication"] is None  # not printed: empty, never guessed
    for left in ("string", ""):  # what the /docs page sends for a box left alone: treated as not sent
        same = client.post("/v1/read-label", files={"image": ("p.png", photo)}, data={"medications": left})
        assert same.status_code == 200 and same.json()["candidates"][0]["list_check"] is None
    docs = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))],
                       data={"boxes": "string", "imprint": "string", "shape": "string", "colour": "string"})
    assert docs.status_code == 200 and docs.json()["candidates"][0]["pill_id"] == "R_1"
    pdf = text_pdf(["EXAMPLE CLINIC", "GLUCOPHAGE 500MG TABLET", "Take 1 tablet daily"])
    body = client.post("/v1/read-label", files={"image": ("rx.pdf", pdf, "application/pdf")}).json()
    assert body["source"] == "pdf_text" and body["candidates"][0]["list_check"] is None
    assert body["candidates"][0]["fields"]["directions"]["value"] == "Take 1 tablet daily"


def test_agent_tools_follow_the_api():
    tools = {t["name"]: t for t in TestClient(app).get("/v1/tools").json()["tools"]}
    assert {"detect_pills", "embed_pill", "match_gallery", "identify_pill", "extract_medication_document",
            "find_medication_product", "screen_medication_list", "retrieve_medication_info"} <= set(tools)
    assert tools["match_gallery"]["call"] == {"method": "POST", "path": "/v1/match", "content_type": "application/json"}
    assert "image reference" in tools["embed_pill"]["input_schema"]["properties"]["image"]["description"]
    assert tools["match_gallery"]["output_schema"] == {"$ref": "#/components/schemas/MatchResponse"}


def test_text_reading_down_is_reported_never_silently_skipped(stage2, monkeypatch):
    client, mat_photo = stage2
    monkeypatch.setenv("EPILLS_OCR", "0")
    app.state.reader = None
    red = png(Image.new("RGB", (64, 64), RED))
    body = client.post("/v1/identify", files=[("images", ("p.png", red))]).json()
    assert body["text_reading"] == "unavailable" and any("couldn't be read" in n for n in body["notes"])
    e = client.post("/v1/embed", files={"image": ("p.png", red)}, data={"read_imprint": "true"}).json()
    assert e["text_reading"] == "unavailable" and e["imprint_reads"] is None
    fake = FakeEmbedder()
    gallery = [{"medication_id": "m1", "name": "Red pill", "imprint": "RX;55",
                "embeddings": fake([Image.new("RGB", (9, 9), RED)]).tolist()}]
    scan = client.post("/v1/scan", files={"image": ("p.png", png(mat_photo([((300, 300), (80, 50), RED)])))},
                       data={"gallery": json.dumps(gallery), "model_version": "fake-1"}).json()
    assert scan["text_reading"] == "unavailable" and scan["pills"][0]["imprint_check"] == "unavailable"
    plain = client.post("/v1/embed", files={"image": ("p.png", red)}).json()
    assert plain["text_reading"] == "not_requested"


def test_embeddings_are_repeatable_and_bad_numbers_are_refused(stage2, monkeypatch):
    client, _ = stage2
    red = png(Image.new("RGB", (64, 64), RED))
    a, b = (client.post("/v1/embed", files={"image": ("p.png", red)}).json() for _ in range(2))
    assert a["embedding"] == b["embedding"] and a["dim"] == len(a["embedding"])
    assert np.isfinite(a["embedding"]).all() and abs(np.linalg.norm(a["embedding"]) - 1) < 1e-3
    monkeypatch.setattr(FakeEmbedder, "forward", lambda self, x: np.full((len(x), 8), np.nan, np.float32))
    r = client.post("/v1/embed", files={"image": ("p.png", red)})
    assert r.status_code == 500 and r.json()["code"] == "model_error"


def test_blisters_and_packages_are_refused_not_matched(stage2):
    client, mat_photo = stage2
    dark = png(Image.new("RGB", (400, 300), (20, 20, 25)))  # the fake reader captions it as a blister pack
    body = client.post("/v1/identify", files=[("images", ("p.png", dark))]).json()
    assert body["candidates"] == [] and body["input_check"]["type"] == "blister" and body["input_check"]["abstained"]
    det = client.post("/v1/detect", files={"image": ("p.png", dark)}).json()
    assert det["pill_count"] == 0 and "not_loose_pills" in det["flags"] and "blister" in det["message"]
    scan = client.post("/v1/scan", files={"image": ("p.png", dark)}).json()
    assert scan["pills"] == [] and scan["input_check"]["abstained"]
    said = client.post("/v1/identify", files=[("images", ("p.png", png(Image.new("RGB", (64, 64), RED))))],
                       data={"input_type": "package"}).json()
    assert said["candidates"] == [] and said["input_check"] == {**said["input_check"], "type": "package",
                                                                 "checked": False, "abstained": True}
    off = client.post("/v1/identify", files=[("images", ("p.png", dark))], data={"check_input": "false"}).json()
    assert off["candidates"] and off["input_check"]["type"] == "unchecked"
    pills = client.post("/v1/detect", files={"image": ("p.png", png(mat_photo([((300, 300), (80, 50), RED)])))}).json()
    assert pills["input_check"]["type"] == "loose_pills" and pills["pill_count"] == 1


def test_check_input_routes_a_photo_to_the_tools_that_fit(stage2):
    client, mat_photo = stage2
    dark = png(Image.new("RGB", (400, 300), (20, 20, 25)))  # the fake reader captions it as a blister pack
    blister = client.post("/v1/check-input", files={"image": ("p.png", dark)}).json()
    assert blister["input_check"]["type"] == "blister" and blister["next_tools"] == [] and "label" in blister["message"]
    loose = client.post("/v1/check-input", files={"image": ("p.png", png(mat_photo([((300, 300), (80, 50), RED)])))})
    assert loose.json()["input_check"]["type"] == "loose_pills" and "match_gallery" in loose.json()["next_tools"]
    said = client.post("/v1/check-input", files={"image": ("p.png", dark)}, data={"input_type": "document"}).json()
    assert said["next_tools"] == ["extract_medication_document"] and not said["input_check"]["checked"]
    tools = {t["name"] for t in client.get("/v1/tools").json()["tools"]}
    assert "route_medication_image" in tools and all(n in tools for r in (loose.json(), said) for n in r["next_tools"])


def test_reading_that_would_be_too_slow_on_cpu_is_skipped_and_said(stage2):
    import serving.app.main as main
    client, _ = stage2
    red = png(Image.new("RGB", (64, 64), RED))
    client.post("/v1/identify", files=[("images", ("p.png", red))])
    main.vision().seconds_per_image = 10.0  # as measured on a slow CPU: one pill's reading would take 40 s
    body = client.post("/v1/identify", files=[("images", ("p.png", red))], data={"check_input": "false"}).json()
    assert body["text_reading"] == "skipped" and any("too long" in n for n in body["notes"])
    assert body["candidates"]  # the visual answer still comes back
