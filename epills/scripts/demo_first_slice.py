"""
The API side of the blueprint's first slice, run end to end with a synthetic patient (evidence for judging):
manual enrolment -> one loose pill at dose time -> /v1/embed + /v1/match -> reference dose rules -> the patient's
answer -> a dose event with vision_status and intake_status kept apart. What the app's backend and agent would do
around the calls (regimen store, the deterministic gate's decision, asking the patient) is simulated and marked so.

The patient's medicines are C3PI validation medicines (never trained on): metformin, lisinopril, amlodipine and
simvastatin, 3 enrolment photos each. Dose 1: an unseen photo of their metformin. Dose 2: a warfarin pill that isn't
theirs (expected: unknown, then catalogue candidates from the same embedding, then "set it aside").

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\demo_first_slice.py                  in-process, with the local model (CPU)
  .venv\\Scripts\\python scripts\\demo_first_slice.py --base https://davidpansx-pharmascan-api.hf.space
Output: outputs/evidence/first_slice_trace[_<host>].json (every call: endpoint, status, seconds, key answer fields;
no embeddings or photos) and a printed summary.
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from PIL import Image

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EPILLS_ROOT / "src"), str(EPILLS_ROOT)]
RAW = EPILLS_ROOT / "data/raw/c3pi/images"
PATIENT = {  # medication id (the app's pseudonymous id) -> C3PI pill type
    "pm-metformin": "627560143_1", "pm-lisinopril": "001723758_1", "pm-amlodipine": "000691540_1",
    "pm-simvastatin": "658620052_1",
}
NOT_THEIRS = "683820052_1"  # warfarin 1 mg (Taro)
MAX_SIDE = 2000  # photos are sent at most this size, like a phone app would


class Trace:
    def __init__(self, client):
        self.client, self.calls = client, []

    def __call__(self, step, method, path, keep=(), **kw):
        t = time.time()
        r = getattr(self.client, method)(path, **kw)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        self.calls.append({"step": step, "call": f"{method.upper()} {path}", "status": r.status_code,
                           "seconds": round(time.time() - t, 2),
                           "answer": {k: body.get(k) for k in keep} if isinstance(body, dict) else None})
        if r.status_code != 200:
            raise SystemExit(f"{step}: {method.upper()} {path} -> {r.status_code} {r.text[:300]}")
        return body


def photos():
    m = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    crops = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/crops/crops.csv", dtype=str, keep_default_na=False)
    m = m.merge(crops.drop_duplicates("crop_path")[["crop_path", "width", "height", "x0", "y0", "x1", "y1"]],
                on="crop_path")
    m["pill_id"] = m.ndc9 + "_" + m.part
    return {pid: rows.sort_values("crop_path") for pid, rows in m.groupby("pill_id")}


def jpeg(row):
    """The photo as a phone app would send it (upright, at most MAX_SIDE), and the pill's box scaled to it."""
    img = Image.open(RAW / row.path).convert("RGB")
    if img.size != (int(row.width), int(row.height)) and img.size[::-1] == (int(row.width), int(row.height)):
        img = img.transpose(Image.Transpose.ROTATE_90)
    s = min(1.0, MAX_SIDE / max(img.size))
    img = img.resize((round(img.width * s), round(img.height * s)))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue(), [round(float(getattr(row, k)) * s, 1) for k in ("x0", "y0", "x1", "y1")]


def dose(call, label, photo, gallery, version, schedule):
    """One dose check, as the app's dose-check orchestrator would run it (blueprint flow 03)."""
    image = {"image": ("dose.jpg", photo)}
    route = call(f"{label}: what does the photo show", "post", "/v1/check-input", files=image,
                 keep=("input_check", "next_tools"))
    if "match_gallery" not in route["next_tools"]:
        return {"vision_status": "not_checked", "reason": route["message"]}
    det = call(f"{label}: find the pill", "post", "/v1/detect", files={"image": ("dose.jpg", photo)},
               data={"check_input": "false"}, keep=("pill_count", "count_reliable", "flags"))
    if not det["count_reliable"] or det["pill_count"] != 1:
        return {"vision_status": "not_checked", "reason": "count not reliable: ask for a new photo"}
    e = call(f"{label}: embed the pill", "post", "/v1/embed", files={"image": ("dose.jpg", photo)},
             data={"box": json.dumps(det["boxes"][0]), "read_imprint": "true"},
             keep=("model_version", "embedding_space_id", "imprint_reads", "text_reading"))
    res = call(f"{label}: compare with the gallery", "post", "/v1/match",
               json={"queries": [e["embedding"]], "gallery": gallery, "model_version": version,
                     "query_reads": [e["imprint_reads"] or []]},
               keep=("results", "threshold", "margin", "evidence"))["results"][0]
    for k in ("closest", "runner_up"):  # keep the trace short
        res[k] = res[k] and {x: res[k][x] for x in ("medication_id", "similarity")}
    event = {"vision_status": res["decision"], "looks_like": res["medication_id"], "similarity": res["similarity"],
             "imprint_check": res["imprint_check"], "message": res["message"]}
    if res["decision"] == "unknown":
        cand = call(f"{label}: catalogue candidates from the same embedding", "post", "/v1/search-by-embedding",
                    json={"embeddings": [e["embedding"]], "model_version": version,
                          "imprint_reads": e["imprint_reads"] or []},
                    keep=("verdict", "confidence", "notes"))
        event["catalogue"] = {"verdict": cand["verdict"], "confidence": cand["confidence"],
                              "first": cand["candidates"][0]["rxnorm_name"] if cand["candidates"] else None}
    pills = [{"medication_id": res["medication_id"], "decision": res["decision"]}]
    rules = call(f"{label}: reference dose rules", "post", "/v1/safety/dose",
                 json={"schedule": schedule, "pills": pills,
                       "names": {g["medication_id"]: g["name"] for g in gallery}}, keep=("summary",))
    event["alerts"] = [{"type": a["type"], "severity": a["severity"], "message": a["message"],
                        "confirm_count": a["details"].get("confirm_count", False)} for a in rules["alerts"]]
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="the API's address; default: in-process with the local model")
    args = parser.parse_args()
    if args.base:
        import httpx
        client = httpx.Client(base_url=args.base, timeout=180)
        where = urlparse(args.base).hostname
    else:
        from fastapi.testclient import TestClient
        from serving.app.main import app
        client, where = TestClient(app), "local"
    call, pics = Trace(client), photos()

    health = call("start: is the API ready", "get", "/health", keep=("ready", "api_version", "model"))
    info = call("start: served model", "get", "/v1/model-info", keep=("model_version", "embedding_space_id"))
    # enrolment (blueprint flow 02): the patient confirmed these medicines; the app embeds 3 loose-pill photos each
    gallery, version = [], info["model_version"]
    for mid, pid in PATIENT.items():
        product = call(f"enrol {mid}: catalogue product", "get", f"/v1/pills/{pid}",
                       keep=("rxnorm_name", "imprint", "strength"))
        embs = []
        for row in pics[pid].head(3).itertuples():
            photo, box = jpeg(row)
            e = call(f"enrol {mid}: embed photo", "post", "/v1/embed", files={"image": ("p.jpg", photo)},
                     data={"box": json.dumps(box)}, keep=("model_version", "embedding_space_id", "framed_by"))
            embs.append(e["embedding"])
        gallery.append({"medication_id": mid, "name": product["rxnorm_name"], "imprint": product["imprint"],
                        "reference_set_id": f"{mid}-loose-v1", "embeddings": embs})
    meds = [{"id": g["medication_id"], "name": g["name"]} for g in gallery]
    screen = call("enrol: screen the list", "post", "/v1/safety/medications", json={"medications": meds},
                  keep=("coverage", "note"))

    dose1_photo, _ = jpeg(pics[PATIENT["pm-metformin"]].iloc[5])  # not one of the enrolment photos
    dose2_photo, _ = jpeg(pics[NOT_THEIRS].iloc[0])
    events = []
    for label, photo, patient_says in (("dose 1 (their metformin)", dose1_photo, "taken"),
                                       ("dose 2 (a warfarin pill, not theirs)", dose2_photo, "not_taken")):
        ev = dose(call, label, photo, gallery, version, [{"medication_id": "pm-metformin", "count": 1}])
        # simulated: the gate shows the result, the patient answers; intake is their answer, never the vision result
        events.append({"dose": label, **ev, "intake_status": patient_says, "intake_source": "patient (simulated)",
                       "model_version": version, "evidence": "visual"})

    out = EPILLS_ROOT / "outputs/evidence" / ("first_slice_trace.json" if where == "local"
                                              else f"first_slice_trace_{where}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"api": args.base or "in-process", "api_version": health["api_version"],
                               "model_version": version, "patient": PATIENT, "safety_coverage": screen["coverage"],
                               "events": events, "calls": call.calls}, indent=1, default=str), encoding="utf-8")
    print(f"API {health['api_version']}, model {version}; {len(call.calls)} calls, "
          f"{sum(c['seconds'] for c in call.calls):.1f} s in all")
    for ev in events:
        print(f"\n{ev['dose']}: vision {ev['vision_status']} ({ev.get('looks_like')}, similarity "
              f"{ev.get('similarity')}, imprint {ev.get('imprint_check')}); intake {ev['intake_status']}")
        print(f"  {ev.get('message')}")
        if ev.get("catalogue"):
            print(f"  catalogue: {ev['catalogue']}")
        for a in ev.get("alerts", []):
            print(f"  alert {a['type']} ({a['severity']}): {a['message']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
