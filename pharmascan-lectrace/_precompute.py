"""
One-off data export for the lecture (run with the epills environment, not this folder's):

    ..\\epills\\.venv\\Scripts\\python _precompute.py

From the served model (epills/outputs/serving/stage2_model.pt, run 4) it writes to data/:
  weights.npz    the small layers after the backbones: capsule fusion, transformer refinement, embedding head
                 (about 0.6M numbers), so the lecture runs them in numpy and the trace shows their real values
  backbones.npz  the two backbones' outputs (VGG16+CBAM -> 256, ViT-B/16 -> 768) for every crop in the demo, plus
                 the full model's embeddings, to check the numpy forward pass against PyTorch
  catalogue.npz  reference embeddings of the 300 catalogue products nearest the foreign pill (of 4,936)
  demo.json      the patient, the photos, the imprint reads, the catalogue products' imprints and token weights,
                 and the served thresholds
  images/        the photos with the pill finder's box, the crops, and where each backbone looks
The patient and photos are the ones in epills/scripts/demo_first_slice.py (C3PI validation medicines, never trained
on), so the lecture matches the live API's first-slice trace.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
EPILLS = HERE.parent / "epills"
sys.path[:0] = [str(EPILLS / "src"), str(EPILLS / "scripts")]

from demo_first_slice import PATIENT, jpeg, photos  # noqa: E402
from epills.core.catalogue.confidence import HIGH, LOW, T_GOOD, T_POOR, WEIGHTS  # noqa: E402
from epills.core.catalogue.store import IMPRINT_ALPHA, IMPRINT_TOP_K, Catalogue  # noqa: E402
from epills.core.inference.embedder import Embedder  # noqa: E402
from epills.core.inference.pill_finder import detect_pills  # noqa: E402

OUT, IMG = HERE / "data", HERE / "images"
CROPS = EPILLS / "data/interim/c3pi/crops/images"
N_CATALOGUE = 300
# The foreign pill: escitalopram 5 mg (M;EC5), a validation medicine that isn't the patient's. The first slice uses a
# warfarin photo whose imprint the reader can't see ('O' only); this photo's imprint is read ('EC5'), so the lecture
# can show what re-ranking does (found with a search over validation photos: look rank 4 -> 1).
FOREIGN, FOREIGN_PHOTO = "003783855_1", "PillProjectDisc22/images/B64V-IMQT3413PKBKHQ2W17AHHFYY-E.jpg"


def linear(prefix, layer, out):
    out[f"{prefix}.W"], out[f"{prefix}.b"] = layer.weight.detach().numpy(), layer.bias.detach().numpy()


def export_weights(model):
    w = {}
    for i, (mlp, bn) in enumerate(zip(model.fusion.mlps, model.fusion.bns)):
        linear(f"fusion.mlp{i}.0", mlp[0], w), linear(f"fusion.mlp{i}.1", mlp[3], w)
        w[f"fusion.bn{i}.mean"], w[f"fusion.bn{i}.var"] = bn.running_mean.numpy(), bn.running_var.numpy()
        w[f"fusion.bn{i}.gamma"], w[f"fusion.bn{i}.beta"] = bn.weight.detach().numpy(), bn.bias.detach().numpy()
        w[f"fusion.bn{i}.eps"] = np.float32(bn.eps)
    r = model.refine
    linear("refine.proj", r.proj, w)
    w["refine.pos"] = r.pos_embedding.detach().numpy()[0]
    for i, layer in enumerate(r.layers):
        for name in ("q", "k", "v", "out"):
            linear(f"refine.{i}.{name}", getattr(layer.mha, name), w)
        linear(f"refine.{i}.ffn0", layer.ffn[0], w), linear(f"refine.{i}.ffn1", layer.ffn[2], w)
        for n in ("norm1", "norm2"):
            ln = getattr(layer, n)
            w[f"refine.{i}.{n}.g"], w[f"refine.{i}.{n}.b"] = ln.weight.detach().numpy(), ln.bias.detach().numpy()
            w[f"refine.{i}.{n}.eps"] = np.float32(ln.eps)
    linear("head", model.embedding_head, w)
    np.savez_compressed(OUT / "weights.npz", **{k: np.asarray(v, np.float32) for k, v in w.items()})
    return sum(np.asarray(v).size for v in w.values())


@torch.no_grad()
def backbones(model, x):
    """The two backbone outputs, CBAM's spatial attention (7x7) and the ViT's [CLS] attention (14x14)."""
    vgg_maps = model.vgg((x - model.imagenet_mean) / model.imagenet_std)
    cb = model.cbam
    avg = F.relu(cb.avg_dense(vgg_maps.mean(dim=(2, 3))))
    mx = F.relu(cb.max_dense(vgg_maps.amax(dim=(2, 3))))
    ch = vgg_maps * torch.sigmoid(cb.channel_dense(avg + mx))[:, :, None, None]
    spatial = torch.sigmoid(cb.spatial(torch.cat([ch.mean(1, keepdim=True), ch.amax(1, keepdim=True)], 1)))
    vgg = F.relu(model.vgg_dense((ch * spatial).mean(dim=(2, 3))))
    model.vit.set_attn_implementation("eager")  # the fused kernel returns no attention weights
    vit_out = model.vit(pixel_values=(x - 0.5) / 0.5, output_attentions=True)
    cls_attn = vit_out.attentions[-1][:, :, 0, 1:].mean(1).reshape(-1, 14, 14)  # last layer, heads averaged
    return vgg.numpy(), vit_out.pooler_output.numpy(), spatial[:, 0].numpy(), cls_attn.numpy()


def overlay(crop, heat, path):
    """The crop with a heat map (blue, transparent where low) over it."""
    crop = crop.convert("RGB").resize((224, 224))
    h = (heat - heat.min()) / (np.ptp(heat) + 1e-9)
    h = np.asarray(Image.fromarray((h * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)) / 255.0
    base = np.asarray(crop, np.float32)
    blue = np.array([42, 120, 214], np.float32)
    a = (0.65 * h)[..., None]
    Image.fromarray((base * (1 - a) + blue * a).astype(np.uint8)).save(path)


def main():
    OUT.mkdir(exist_ok=True), IMG.mkdir(exist_ok=True)
    emb = Embedder(EPILLS / "outputs/serving/stage2_model.pt", device="cpu")  # float32, as on the CPU path
    model, window = emb.model, emb.window
    cal = emb.calibration
    print("model", emb.version, "window", window, "small-layer numbers:", export_weights(model))

    pics, cat = photos(), Catalogue(EPILLS / "data/processed/catalogue")
    items = []  # (role, medication id, pill id, manifest row)
    for mid, pid in PATIENT.items():
        items += [("enrol", mid, pid, row) for row in pics[pid].head(3).itertuples()]
    items.append(("dose", "pm-metformin", PATIENT["pm-metformin"], next(pics[PATIENT["pm-metformin"]].iloc[[5]]
                                                                       .itertuples())))
    foreign = pics[FOREIGN]
    items.append(("dose", None, FOREIGN, next(foreign[foreign.crop_path == FOREIGN_PHOTO].itertuples())))

    crops, meta = [], []
    reads_cache = json.loads((EPILLS / "data/interim/ocr/large_c92/validation_all.json").read_text())
    for k, (role, mid, pid, row) in enumerate(items):
        photo_bytes, box_given = jpeg(row)
        import io
        photo = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
        det = detect_pills(photo) if role == "dose" else None
        box = det.boxes[0] if det and det.boxes else box_given
        crop = emb.frame(photo, [box])[0]
        crops.append(crop)
        name = f"{k:02d}_{role}_{pid}"
        crop.resize((224, 224)).save(IMG / f"{name}_crop.png")
        entry = {"role": role, "medication_id": mid, "pill_id": pid, "crop": f"images/{name}_crop.png",
                 "imprint": cat.product(pid)["imprint"], "name": cat.product(pid)["rxnorm_name"]}
        if role == "dose":
            shown = photo.copy()
            s = 900 / max(shown.size)
            shown = shown.resize((round(shown.width * s), round(shown.height * s)))
            d = ImageDraw.Draw(shown)
            d.rectangle([v * s for v in box], outline=(235, 104, 52), width=5)
            shown.save(IMG / f"{name}_photo.jpg", quality=85)
            entry.update(photo=f"images/{name}_photo.jpg", box=[round(float(v), 1) for v in box],
                         pills_found=det.pill_count, finder_flags=det.flags,
                         reads=reads_cache.get(str(CROPS / row.crop_path), []))
        meta.append(entry)

    x = torch.stack([emb.transform(c) for c in crops])
    vgg, vit, cbam_maps, vit_maps = backbones(model, x)
    with torch.no_grad():
        full = model(x)["embedding"].numpy()
    np.savez_compressed(OUT / "backbones.npz", vgg=vgg, vit=vit, torch_embedding=full, cnn_gate=cbam_maps, vit_cls_attention=vit_maps)
    for k, e in enumerate(meta):
        if e["role"] == "dose":
            stem = e["crop"].replace("_crop.png", "")
            overlay(crops[k], cbam_maps[k], HERE / f"{stem}_cnn_attention.png")
            overlay(crops[k], vit_maps[k], HERE / f"{stem}_vit_attention.png")
            e["cnn_attention"], e["vit_attention"] = f"{stem}_cnn_attention.png", f"{stem}_vit_attention.png"

    # the catalogue products nearest the foreign pill (by the served look score), with their reference embeddings
    q = full[-1] / np.linalg.norm(full[-1])
    per_product = np.maximum.reduceat(cat.emb @ q, cat.starts)
    near = np.argsort(-per_product)[:N_CATALOGUE]
    rows = np.concatenate([np.arange(cat.starts[i], cat.starts[i + 1] if i + 1 < len(cat.starts) else len(cat.emb))
                           for i in near])
    owner = np.concatenate([np.full((cat.starts[i + 1] if i + 1 < len(cat.starts) else len(cat.emb)) - cat.starts[i],
                                    j) for j, i in enumerate(near)])
    np.savez_compressed(OUT / "catalogue.npz", emb=cat.emb[rows].astype(np.float32), owner=owner)
    index = cat.imprint_index  # token weights are rarities over the whole catalogue
    products = []
    for i in near:
        pid = str(cat.segment_pill[i])
        p = cat.product(pid)
        products.append({"pill_id": pid, "name": p["rxnorm_name"] or p["name"], "imprint": p["imprint"] or "",
                         "tokens": {t: round(index.weight[t], 4) for t in index.tokens[i]}})

    demo = {"model_version": emb.version, "window": window,
            "threshold": cal.get("threshold"), "margin": cal.get("margin"),
            "imprint_alpha": IMPRINT_ALPHA, "imprint_top_k": IMPRINT_TOP_K,
            "confidence": {"weights": WEIGHTS, "t_poor": T_POOR, "t_good": T_GOOD, "low": LOW, "high": HIGH},
            "patient": PATIENT, "items": meta, "catalogue_products": products,
            "catalogue_size": len(cat.segment_pill)}
    (OUT / "demo.json").write_text(json.dumps(demo, indent=1), encoding="utf-8")
    print(json.dumps({k: demo[k] for k in ("model_version", "window", "threshold", "margin")}))
    for e in meta:
        if e["role"] == "dose":
            print(e["name"], "finder:", e["pills_found"], e["finder_flags"], "reads:", e["reads"])


if __name__ == "__main__":
    main()
