"""
Build the pill catalogue for the API (features F1b and F2, docs/features.md): every C3PI pill type we have pictures
of, with names, RxNorm ingredients, display pictures and the embeddings of its crops under one model version.

Steps:
  1. Crop every studio reference photo (MC_CHALLENGE, RxImage; all pill types, not only training ones) with the
     reference cropper -> data/interim/c3pi/catalogue_crops/ (kept between runs).
  2. Add up to --photos-per-product everyday crops per pill type (data/interim/c3pi/crops/).
  3. Embed all crops with the served model (scripts/export_model.py output).
  4. Names: NDC -> RxNorm concept (retired codes too) -> active ingredients; the C3PI names as they are.
  5. Display pictures: RxImage (both sides, with a mm ruler), else the challenge crop, else an everyday crop.
  6. Write data/processed/catalogue/ (catalogue.sqlite, embeddings.npz, images/), then a self-check: everyday photos
     left out of the catalogue are identified against it.

The catalogue belongs to one model version: rebuild it for every new model. C3PI's licence: research and
development use only (docs/datasets.md).

Usage, from the epills folder:  .venv\\Scripts\\python scripts\\build_catalogue.py
"""
import argparse
import datetime
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.catalogue.store import Catalogue, write_catalogue  # noqa: E402
from epills.core.inference.embedder import Embedder  # noqa: E402
from epills.core.safety.rxnorm import RxNormClient  # noqa: E402
from epills.training.datasets.c3pi_ref_crop import crop_challenge, crop_rxnav  # noqa: E402

RAW = EPILLS_ROOT / "data/raw/c3pi/images"
EVERYDAY = EPILLS_ROOT / "data/interim/c3pi/crops"
REF_CROPS = EPILLS_ROOT / "data/interim/c3pi/catalogue_crops"
REF_TYPES = {"MC_CHALLENGE_V1.0": crop_challenge, "MC_API_RXNAV_V1.3": crop_rxnav}


def crop_reference(row):
    path, image_type = row
    stem = Path(path).with_suffix("").as_posix()
    done = sorted(REF_CROPS.glob(f"{stem}_*.jpg"))
    if done:
        return [p.relative_to(REF_CROPS).as_posix() for p in done], "cached"
    img = cv2.imread(str(RAW / path))
    if img is None:
        return [], "unreadable"
    crops, note = REF_TYPES[image_type](img)
    if note != "ok":  # as for training: flagged photos were mostly wrong on review
        return [], note
    out = []
    for n, crop in enumerate(crops):
        dest = REF_CROPS / f"{stem}_{n}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(dest.relative_to(REF_CROPS).as_posix())
    return out, note


def rxnorm_for(client, ndc11, generic_name):
    """(rxcui, concept name, status, [(ingredient rxcui, name)]) from the NDC, else from the generic name."""
    try:
        s = client.ndc_status(ndc11) if ndc11 else None
        if s:
            return s["rxcui"], s["name"], s["status"], client.ingredients(s["rxcui"])
        rxcui = client.rxcui_by_name(generic_name) if generic_name else None
        if rxcui:
            return rxcui, None, "by generic name", client.ingredients(rxcui)
    except Exception as error:
        return None, None, f"lookup failed: {error}", []
    return None, None, "not found", []


def save_display(src, dest, width):
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, round(im.height * width / im.width)), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, quality=82)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=EPILLS_ROOT / "outputs/serving/stage2_model.pt")
    parser.add_argument("--out", type=Path, default=EPILLS_ROOT / "data/processed/catalogue")
    parser.add_argument("--photos-per-product", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    meta = pd.read_csv(EPILLS_ROOT / "data/interim/c3pi/metadata.csv", dtype=str, keep_default_na=False)
    meta = meta[meta.downloaded == "1"].copy()
    meta["pill_id"] = meta.ndc9 + "_" + meta.part

    # 1. reference crops
    refs = meta[meta.image_type.isin(REF_TYPES)].reset_index(drop=True)
    with Pool(12) as pool:
        results = pool.map(crop_reference, list(zip(refs.path, refs.image_type)), chunksize=8)
    ref_rows = [(pid, REF_CROPS / c, "reference") for pid, (crops, _) in zip(refs.pill_id, results) for c in crops]
    print(f"reference crops: {len(ref_rows):,} from {len(refs):,} photos "
          f"({sum(1 for c, n in results if not c)} photos skipped)", flush=True)

    # 2. everyday crops: some go into the catalogue, the rest are kept for the self-check
    crops = pd.read_csv(EVERYDAY / "crops.csv", dtype=str, keep_default_na=False)
    crops = crops[(crops.found == "1") & (crops.crop_path != "")].copy()
    crops["pill_id"] = crops.ndc9 + "_" + crops.part
    crops = crops.sample(frac=1, random_state=args.seed)
    crops["rank"] = crops.groupby("pill_id").cumcount()
    in_cat = crops[crops["rank"] < args.photos_per_product]
    held_out = crops[crops["rank"] >= args.photos_per_product].groupby("pill_id").head(2)
    photo_rows = [(pid, EVERYDAY / "images" / p, "photo") for pid, p in zip(in_cat.pill_id, in_cat.crop_path)]

    # 3. embeddings
    embedder = Embedder(args.model)
    rows = ref_rows + photo_rows
    emb = []
    for i in range(0, len(rows), 256):
        emb.append(embedder([Image.open(p) for _, p, _ in rows[i:i + 256]]))
        print(f"\rembedded {min(i + 256, len(rows)):,} / {len(rows):,}", end="", flush=True)
    emb = np.concatenate(emb)
    print(flush=True)

    # 4. products and names
    first = meta.sort_values("image_type").groupby("pill_id").first()
    ndc11 = meta[meta.ndc11 != ""].groupby("pill_id").ndc11.agg(lambda s: s.value_counts().index[0])
    pill_ids = sorted(set(r[0] for r in rows))
    client = RxNormClient(cache_dir=EPILLS_ROOT / "data/safety/rxnorm_cache")
    with ThreadPoolExecutor(12) as pool:
        names = list(pool.map(lambda pid: rxnorm_for(client, ndc11.get(pid), first.at[pid, "generic_name"]), pill_ids))
    products, ingredients = [], []
    for pid, (rxcui, rx_name, status, ings) in zip(pill_ids, names):
        f = first.loc[pid]
        products.append({
            "pill_id": pid, "ndc9": f.ndc9, "part": int(f.part or 1), "parts": int(f.parts or 1),
            "ndc11": ndc11.get(pid), "name": f.proprietary_name, "generic_name": f.generic_name,
            "manufacturer": f.labeled_by, "shape": f["shape"], "colour": f.color,
            "size_mm": float(f.size_mm) if f.size_mm else None, "imprint": f.imprint, "imprint_type": f.imprint_type,
            "imprint_colour": f.imprint_color, "score": int(f.score) if f.score.isdigit() else None,
            "rxcui": rxcui, "rxnorm_name": rx_name, "rxnorm_status": status})
        ingredients += [(pid, r, n) for r, n in ings]
    print(f"RxNorm: {sum(1 for n in names if n[3]):,} of {len(pill_ids):,} products have ingredients", flush=True)

    # 5. display pictures
    images = []
    img_dir = args.out / "images"
    by_pid = {}
    for _, r in meta[meta.image_type.isin(REF_TYPES)].sort_values("image_type").iterrows():
        by_pid.setdefault(r.pill_id, []).append(r)
    for pid in pill_ids:
        n = 0
        for r in by_pid.get(pid, [])[:3]:
            if r.image_type == "MC_API_RXNAV_V1.3":
                src, kind, width = RAW / r.path, "rximage", 512
            else:
                done = sorted(REF_CROPS.glob(f"{Path(r.path).with_suffix('').as_posix()}_*.jpg"))
                if not done:
                    continue
                src, kind, width = done[0], "reference", 320
            file = f"{pid}_{n}.jpg"
            save_display(src, img_dir / file, width)
            images.append((pid, n, kind, file))
            n += 1
        if n == 0:
            mine = in_cat[in_cat.pill_id == pid]
            if len(mine):
                file = f"{pid}_0.jpg"
                save_display(EVERYDAY / "images" / mine.crop_path.iloc[0], img_dir / file, 320)
                images.append((pid, 0, "photo", file))

    write_catalogue(args.out, products, ingredients, images, emb, [r[0] for r in rows], [r[2] for r in rows], {
        "model_version": embedder.version, "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "NLM C3PI (research and development use only); names: RxNorm (NLM)",
        "products": len(products), "embeddings": int(len(emb)),
        "photos_per_product": args.photos_per_product})
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {args.out}: {len(products):,} products, {len(emb):,} embeddings, {len(images):,} pictures "
          f"({size:.0f} MB)", flush=True)

    # 6. self-check: everyday photos not in the catalogue, identified against it
    cat = Catalogue(args.out)
    splits = pd.read_csv(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv", dtype=str, keep_default_na=False)
    split_of = splits.assign(pill_id=splits.ndc9 + "_" + splits.part).groupby("pill_id").split.first()
    q = held_out[held_out.pill_id.isin(cat.attrs)]
    q_emb = embedder([Image.open(EVERYDAY / "images" / p) for p in q.crop_path])
    hits = []
    for e, pid in zip(q_emb, q.pill_id):
        top = [c["pill_id"] for c in cat.identify(e[None], top_k=5)[0]]
        hits.append((split_of.get(pid, "other"), top[:1] == [pid], pid in top))
    hits = pd.DataFrame(hits, columns=["split", "top1", "top5"])
    print("self-check, new everyday photos of catalogue products (types the model trained on are optimistic):")
    print(hits.groupby("split").agg(photos=("top1", "size"), top1=("top1", "mean"), top5=("top5", "mean")).round(3))


if __name__ == "__main__":
    main()
