"""
Response times of the hosted API, end to end from this computer (target: every call under 5 s once the Space is awake).

Each endpoint is called --repeats times with realistic inputs (a pill crop, a photo of several pills, a dispensing
label, a prescription PDF); the first call per endpoint is reported separately (it may start a GPU worker).

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\benchmark_latency.py [--base https://davidpansx-pharmascan-api.hf.space] [--repeats 5]
Output: printed table, and outputs/latency/<timestamp>.json
"""
import argparse
import datetime
import io
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "tests"))
CROP = EPILLS_ROOT / "data/interim/c3pi/catalogue_crops/PillProjectDisc1/images/!!XUD84WEVV5WVC6O1OD9DS0!DBV6X_0.jpg"


def jpeg(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def several_pills():
    """A plain mat with three pills cut from catalogue photos (like a dose photo)."""
    mat = Image.new("RGB", (1600, 1200), (205, 200, 195))
    crop = Image.open(CROP).convert("RGB").resize((260, 260))
    for x, y in ((250, 300), (800, 520), (1200, 250)):
        mask = Image.new("L", crop.size, 0)
        ImageDraw.Draw(mask).ellipse((40, 40, 220, 220), fill=255)
        mat.paste(crop, (x, y), mask)
    return mat


def label():
    img = Image.new("RGB", (900, 420), "white")
    d = ImageDraw.Draw(img)
    try:
        font = lambda s: ImageFont.truetype("C:/Windows/Fonts/arial.ttf", s)  # noqa: E731
    except OSError:
        font = lambda s: ImageFont.load_default()  # noqa: E731
    y = 30
    for text, size in (("EXAMPLE POLYCLINIC", 26), ("PARACETAMOL 500MG TABLET", 40), ("Take 2 tablets 4 times a day", 30),
                       ("when necessary for pain or fever", 30), ("Qty: 20 TAB", 26)):
        d.text((40, y), text, fill=(20, 20, 30), font=font(size))
        y += int(size * 1.35)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="https://davidpansx-pharmascan-api.hf.space")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    from test_serving import text_pdf

    crop, photo, lbl = CROP.read_bytes(), jpeg(several_pills()), jpeg(label())
    pdf = text_pdf(["EXAMPLE CLINIC", "1. Metformin 500 mg tablet", "Take 1 tablet twice a day after meals",
                    "Qty: 60 tablets", "2. Amlodipine 5 mg tablet", "Take 1 tablet once daily"])
    client = httpx.Client(base_url=args.base, timeout=120)
    emb = client.post("/v1/embed", files={"image": ("p.jpg", crop)}).json()
    version, vec = emb["model_version"], emb["embedding"]
    gallery = [{"medication_id": "m1", "name": "Metformin 500 mg", "embeddings": [vec], "imprint": "BMS;6060;500"},
               {"medication_id": "m2", "name": "Other", "embeddings": [[-v for v in vec]]}]
    calls = {
        "GET /health": lambda: client.get("/health"),
        "POST /v1/embed": lambda: client.post("/v1/embed", files={"image": ("p.jpg", crop)}),
        "POST /v1/embed +imprint": lambda: client.post("/v1/embed", files={"image": ("p.jpg", crop)},
                                                       data={"read_imprint": "true"}),
        "POST /v1/match": lambda: client.post("/v1/match", json={"queries": [vec], "gallery": gallery,
                                                                 "model_version": version}),
        "POST /v1/identify": lambda: client.post("/v1/identify", files=[("images", ("p.jpg", crop))]),
        "POST /v1/detect (3 pills)": lambda: client.post("/v1/detect", files={"image": ("d.jpg", photo)}),
        "POST /v1/scan (3 pills, gallery)": lambda: client.post(
            "/v1/scan", files={"image": ("d.jpg", photo)},
            data={"gallery": json.dumps(gallery), "model_version": version}),
        "POST /v1/scan (3 pills, catalogue)": lambda: client.post("/v1/scan", files={"image": ("d.jpg", photo)}),
        "POST /v1/read-label (photo)": lambda: client.post("/v1/read-label", files={"image": ("l.jpg", lbl)}),
        "POST /v1/read-label (PDF)": lambda: client.post("/v1/read-label", files={"image": ("rx.pdf", pdf)}),
        "POST /v1/safety/medications": lambda: client.post("/v1/safety/medications", json={"medications": [
            {"id": "a", "name": "Lipitor 10 mg"}, {"id": "b", "name": "Clarithromycin 500 mg"}]}),
        "GET /v1/pills/search": lambda: client.get("/v1/pills/search", params={"name": "metformin"}),
    }
    rows = {}
    for name, call in calls.items():
        times, codes = [], set()
        for _ in range(args.repeats + 1):
            t = time.time()
            r = call()
            times.append(time.time() - t)
            codes.add(r.status_code)
        warm = times[1:]
        rows[name] = {"first_s": round(times[0], 2), "median_s": round(statistics.median(warm), 2),
                      "max_s": round(max(warm), 2), "status": sorted(codes)}
        flag = "" if max(warm) < 5 else "  <-- over 5 s"
        print(f"{name:36s} first {times[0]:5.2f}s  median {statistics.median(warm):5.2f}s  max {max(warm):5.2f}s  "
              f"{sorted(codes)}{flag}", flush=True)
    out = EPILLS_ROOT / "outputs/latency" / f"{datetime.datetime.now():%Y%m%d_%H%M}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    health = client.get("/health").json()
    out.write_text(json.dumps({"base": args.base, "api_version": health.get("api_version"),
                               "device": health.get("model", {}).get("device"), "repeats": args.repeats,
                               "calls": rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
