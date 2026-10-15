"""
Singapore-style dispensing stickers, read end to end (evidence for test case P01): two stickers with made-up patients
are rendered, clean and photo-like (on a table, turned 4 degrees, slightly blurred), and sent through /v1/read-label
in-process with the local reader. Prints each medicine's parts and checks that no patient line comes back.

Usage, from the epills folder:  .venv\\Scripts\\python scripts\\check_dispensing_labels.py
Output: outputs/evidence/dispensing_labels/*.png and results.json
"""
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EPILLS_ROOT / "src"), str(EPILLS_ROOT)]
OUT = EPILLS_ROOT / "outputs/evidence/dispensing_labels"
FONT = "C:/Windows/Fonts/arialbd.ttf"
PATIENTS = ("TAN AH KOW", "LIM MEI LING", "TAN AH SENG", "SXXXX123A")  # made up, and MOH's example
STICKERS = {
    "polyclinic": ["EXAMPLE POLYCLINIC PHARMACY", "Name: TAN AH KOW            Date: 14/10/2026",
                   "METFORMIN 500MG TABLET                 Qty: 60 TAB",
                   "Take ONE tablet TWICE a day AFTER FOOD", "For diabetes",
                   "Swallow whole. Do not crush.", "Exp: 10/2027"],
    "retail": ["EXAMPLE PHARMACY @ TAMPINES", "Patient: LIM MEI LING      14 Oct 2026",
               "AMLODIPINE 5MG TABLET", "Quantity: 30 tablets",
               "Take 1 tablet once a day in the morning", "with or without food",
               "For high blood pressure", "May cause dizziness. Avoid grapefruit juice.", "Exp 09/2027"],
}


def render(lines, photo=False):
    try:
        font = ImageFont.truetype(FONT, 30)
    except OSError:
        font = ImageFont.load_default(30)
    img = Image.new("RGB", (900, 70 + 52 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([5, 5, img.width - 6, img.height - 6], outline="black", width=3)
    for k, line in enumerate(lines):
        d.text((30, 35 + 52 * k), line, fill="black", font=font)
    if photo:  # like a phone photo: on a table, slightly turned and soft
        bg = Image.new("RGB", (img.width + 200, img.height + 200), (186, 170, 150))
        bg.paste(img, (100, 100))
        img = bg.rotate(4, expand=True, fillcolor=(186, 170, 150)).filter(ImageFilter.GaussianBlur(1.2))
    return img


MOH_GUIDE = EPILLS_ROOT / "docs/know-about-medication-label_eng.pdf"  # MOH's "Know about medication label"


def moh_label():
    """The example label in MOH's guide (the standard Singapore layout), cut out of the poster; None without it."""
    if not MOH_GUIDE.exists():
        return None
    import pypdfium2
    page = pypdfium2.PdfDocument(str(MOH_GUIDE))[0].render(scale=2).to_pil().convert("RGB")
    return page.crop((210, 638, 985, 1050))


def main():
    from fastapi.testclient import TestClient
    from serving.app.main import app
    client, results = TestClient(app), {}
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [(f"{name}{'_photo' if photo else ''}", lines, photo) for name, lines in STICKERS.items()
             for photo in (False, True)]
    if moh_label() is not None:
        cases.insert(0, ("moh_example", None, False))
    for key, lines, photo in cases:
        img = moh_label() if lines is None else render(lines, photo)
        img.save(OUT / f"{key}.png")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        body = client.post("/v1/read-label", files={"image": ("s.png", buf.getvalue())}).json()
        results[key] = body
        print(f"\n== {key}: label_kind {body.get('label_kind')}, reader {body.get('ocr_model')}")
        for c in body.get("candidates", []):
            f, i = c["fields"], c["fields"]["instructions"] or {}
            print(f"  {c['name']} | {c['rxnorm_name']} | quantity "
                  f"{f['quantity'] and (f['quantity']['amount'], f['quantity']['unit'])}")
            print(f"  instructions {i.get('text')!r}: dose {i.get('dose')}, times a day {i.get('times_per_day')}, "
                  f"every {i.get('every_hours')} h, as needed {i.get('as_needed')}, {i.get('times_of_day')}, "
                  f"food {i.get('food')}")
            print(f"  for {f['indication'] and f['indication']['value']!r} | cautions "
                  f"{[x['value'] for x in f['cautions']]} | date {f['date'] and f['date']['value']} | "
                  f"expiry {f['expiry_date'] and f['expiry_date']['value']}")
        leaked = [p for p in PATIENTS if p in json.dumps(body)]
        print(f"  patient lines returned: {leaked or 'none'}")
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
