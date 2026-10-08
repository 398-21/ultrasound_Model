"""
P6: evaluate a trained, calibrated Stage 2 model on the test sets. Nothing here is tuned on these sets: the model,
the crop framing (a square window 1.4x the pill's long side, the C3PI cropper's rule), the threshold and the margin
(calibration.json, from validation) are all fixed before the test sets are scored.

Test sets (--sets, default all three):
  c3pi_holdout  C3PI everyday photos of the ePillID holdout pill types (manifest split epillid_holdout), never used
                in training or validation. The app scenario: patients with 10 (and 5, 15) medications enrolled
                from 3 photos; unknown pills are photos of the held-out types outside the patient's list. At the
                fixed threshold and margin, with 95% intervals from resampling medications.
  epillid       The ePillID benchmark as published: fold 4 consumer photos as simulated front + back pairs against a
                gallery of the reference photos of all 4,902 pill types; MAP, GAP, MAP@1, GAP@1 as in the ePillID
                code (metrics.py, metric_test_eval.py; side pairs averaged, "post_mean"). Also single photos. Our
                model never trained on reference photos (decision D7): this is everyday-to-studio transfer.
  mediseg       MEDISEG 32-pill set (Hong Kong products, other phones, clutter, small pills): closed set over the
                32 types and the app scenario, enrolment from single-pill photos only; by pill size. Plus the
                MEDISEG-Deploy few-shot set (8 products, 1/5/10 enrolment photos, 40 pills in 5 mixed photos).
Outputs: <run>/eval/<set>.json (and the embeddings, <set>_embeddings.npz) and a printed summary.

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\evaluate_stage2.py --run outputs\\runs\\stage2_run2
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.inference.matching import normalise  # noqa: E402
from epills.core.models.hybrid import HybridPillNet  # noqa: E402
from epills.training.datasets.c3pi_crop import square_crop  # noqa: E402
from epills.training.datasets.pills import PillCrops, Vocab, eval_transform, load_manifest  # noqa: E402
from epills.training.eval.openset import bootstrap, closed_set_top1, operating_point, simulate  # noqa: E402
from epills.training.eval.retrieval import embed  # noqa: E402

WINDOW = 1.4          # crop window / pill long side, as in the C3PI cropper (find_pill)
CROP_SIZE = 384       # as the C3PI crops
K_ENROL = 3
GALLERIES = 1000
EPILLID = EPILLS_ROOT / "data/raw/epillid/ePillID_data"
MEDISEG = EPILLS_ROOT / "data/raw/mediseg"


class Crops(Dataset):
    """
    Pill crops framed like the C3PI crops. items: (image path, box (x0, y0, x1, y1) or None for "the pill fills
    the image"). Pre-cropped images (box None) are padded with their border colour; boxes in full photos are
    cropped from the photo (edges replicated only outside it).
    """

    def __init__(self, items):
        self.items, self.transform = items, eval_transform()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, box = self.items[i]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        if box is None:
            ring = np.concatenate([img[:2].reshape(-1, 3), img[-2:].reshape(-1, 3), img[:, :2].reshape(-1, 3),
                                   img[:, -2:].reshape(-1, 3)])
            fill = tuple(float(v) for v in np.median(ring, axis=0))
            side = WINDOW * max(h, w)
            f = CROP_SIZE / side
            M = np.float32([[f, 0, CROP_SIZE / 2 - w / 2 * f], [0, f, CROP_SIZE / 2 - h / 2 * f]])
            crop = cv2.warpAffine(img, M, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        else:
            crop = square_crop(img, box, WINDOW * max(box[2] - box[0], box[3] - box[1]), CROP_SIZE)
        return {"image": self.transform(Image.fromarray(crop[:, :, ::-1].copy())), "label": i}


def embed_items(model, items, device, cache):
    if cache.exists():
        return np.load(cache)["emb"]
    emb, _ = embed(model, DataLoader(Crops(items), batch_size=128, num_workers=8), device)
    np.savez(cache, emb=emb)
    return emb


def open_set_report(emb, lab, cal, sizes=(5, 10, 15), enrol_mask=None, seed=0, n_boot=1000):
    """The app scenario at the fixed threshold and margin; unknown pills from the medications outside each gallery."""
    out = {}
    for size in sizes:
        known, unknown = simulate(emb, lab, None, size, K_ENROL, GALLERIES, 200, seed, enrol_mask=enrol_mask)
        point = operating_point(known, unknown, cal["threshold"], cal["margin"])
        out[f"gallery_{size}"] = {"at_calibration": point,
                                  "ci95": bootstrap(known, unknown, cal["threshold"], cal["margin"], n=n_boot),
                                  "margin_0": operating_point(known, unknown, cal["threshold"], 0.0)}
    return out


# ---- ePillID metrics, as defined in the ePillID code (metrics.py) -------------------------------------------------
def average_precision(labels, scores):
    """sklearn.metrics.average_precision_score for one binary problem: sum over distinct thresholds of
    (recall step) x precision."""
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order].astype(np.float64)
    last = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]  # last index of each distinct score
    tp = np.cumsum(labels)[last]
    fp = (last + 1) - tp
    precision, recall = tp / (tp + fp), tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def epillid_metrics(sim, correct):
    """sim (queries, classes) similarities; correct (queries,) index of the right class."""
    rows = np.arange(len(sim))
    right = sim[rows, correct]
    rank = 1 + (sim > right[:, None]).sum(axis=1)  # ties count in the query's favour, as argsort does in practice
    onehot = np.zeros(sim.shape, bool)
    onehot[rows, correct] = True
    top = sim.argmax(axis=1)
    return {"MAP": float(np.mean(1.0 / rank)), "MAP@1": float(np.mean(rank == 1)),
            "GAP": average_precision(onehot.ravel(), sim.ravel()),
            "GAP@1": average_precision(top == correct, sim[rows, top]),
            "top5": float(np.mean(rank <= 5)), "queries": int(len(sim))}


def eval_epillid(model, device, out_dir):
    labels = pd.read_csv(EPILLID / "all_labels.csv")
    fold4 = pd.read_csv(EPILLID / "folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/"
                                  "pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv")
    refs = labels[labels.is_ref].reset_index(drop=True)
    root = EPILLID / "classification_data"
    ref_emb = normalise(embed_items(model, [(root / p, None) for p in refs.image_path], device,
                                    out_dir / "epillid_ref_embeddings.npz"))
    q_emb = normalise(embed_items(model, [(root / p, None) for p in fold4.image_path], device,
                                  out_dir / "epillid_fold4_embeddings.npz"))
    classes = np.array(sorted(refs.pilltype_id.unique()))
    ref_cls = np.searchsorted(classes, refs.pilltype_id.to_numpy())
    q_cls = np.searchsorted(classes, fold4.pilltype_id.to_numpy())
    assert (classes[q_cls] == fold4.pilltype_id.to_numpy()).all(), "a holdout type has no reference photo"

    # class score of each photo: best similarity to any reference photo of the class (front or back)
    pw = torch.from_numpy(q_emb @ ref_emb.T)
    cls_sim = torch.full((len(q_emb), len(classes)), -2.0).scatter_reduce(
        1, torch.from_numpy(ref_cls).expand(len(q_emb), -1), pw, reduce="amax").numpy()

    # simulated two-sided queries: every pair of photos of one type with different sides, sides averaged
    front = fold4.is_front.to_numpy().astype(bool)
    pairs = [(i, j) for i in range(len(fold4)) for j in range(i + 1, len(fold4))
             if q_cls[i] == q_cls[j] and front[i] != front[j]]
    pairs = np.array(pairs)
    two_sided = (cls_sim[pairs[:, 0]] + cls_sim[pairs[:, 1]]) / 2
    result = {
        "protocol": "ePillID fold 4; gallery = reference photos of all 4,902 pill types; crops padded to the "
                    "C3PI framing (window 1.4x the image)",
        "two_sided": {**epillid_metrics(two_sided, q_cls[pairs[:, 0]]),
                      "photos": int(len(np.unique(pairs))), "pill_types": int(len(np.unique(q_cls[pairs[:, 0]])))},
        "single_photo": epillid_metrics(cls_sim, q_cls),
        "paper_best_two_sided": {"MAP": 0.9576, "GAP": 0.8120, "MAP@1": 0.9201, "GAP@1": 0.9119},
    }
    return result


# ---- MEDISEG --------------------------------------------------------------------------------------------------------
def coco_items(ann_path, images_dir):
    a = json.load(open(ann_path, encoding="utf-8"))
    images = {i["id"]: i["file_name"] for i in a["images"]}
    names = {c["id"]: c["name"] for c in a["categories"]}
    per_image = pd.Series([x["image_id"] for x in a["annotations"]]).value_counts()
    rows = []
    for x in a["annotations"]:
        bx, by, bw, bh = x["bbox"]
        rows.append({"path": images_dir / images[x["image_id"]], "box": (bx, by, bx + bw, by + bh),
                     "label": names[x["category_id"]], "long_px": max(bw, bh),
                     "single": per_image[x["image_id"]] == 1})
    return pd.DataFrame(rows)


def eval_mediseg(model, device, out_dir, cal):
    items = coco_items(MEDISEG / "MEDISEG/32pills/annotations.json", MEDISEG / "MEDISEG/32pills/images")
    emb = embed_items(model, list(zip(items.path, items.box)), device, out_dir / "mediseg_embeddings.npz")
    lab = items.label.to_numpy()
    single = items.single.to_numpy()

    closed = closed_set_top1(emb, lab, K_ENROL, seed=0, enrol_mask=single)
    q = items.iloc[closed.pop("query_index")].assign(correct=closed.pop("correct_per_query"))
    q["size"] = pd.cut(q.long_px, [0, 50, 100, 10_000], labels=["under 50 px", "50-100 px", "over 100 px"])
    by_size = {str(k): {"top1": float(g.correct.mean()), "queries": int(len(g))}
               for k, g in q.groupby("size", observed=True)}
    q["scene"] = np.where(q.single, "single pill", "several pills")
    by_scene = {k: {"top1": float(g.correct.mean()), "queries": int(len(g))} for k, g in q.groupby("scene")}

    result = {
        "protocol": "MEDISEG 32pills; crops from the box, window 1.4x the long side; enrolment: 3 single-pill "
                    "photos per product; all other pills are queries",
        "pill_long_side_px_quantiles": dict(zip(["5%", "25%", "50%", "75%", "95%"],
                                                np.quantile(items.long_px, [.05, .25, .5, .75, .95]).round(0).tolist())),
        "closed_set_32": {**closed, "by_pill_size": by_size, "by_scene": by_scene},
        "open_set": open_set_report(emb, lab, cal, sizes=(5, 10), enrol_mask=single),
        "deploy": eval_mediseg_deploy(model, device, out_dir, cal),
    }
    return result


def eval_mediseg_deploy(model, device, out_dir, cal):
    root = MEDISEG / "deploy"
    query = coco_items(root / "query/query_annotations.coco.json", root / "query/images")
    q_emb = normalise(embed_items(model, list(zip(query.path, query.box)), device,
                                  out_dir / "mediseg_deploy_query_embeddings.npz"))
    out = {}
    for shots in (1, 5, 10):
        enrol = coco_items(root / f"{shots}-shot/{shots}-shot_annotations.coco.json", root / f"{shots}-shot/images")
        e_emb = normalise(embed_items(model, list(zip(enrol.path, enrol.box)), device,
                                      out_dir / f"mediseg_deploy_{shots}shot_embeddings.npz"))
        names = sorted(enrol.label.unique())
        protos = np.stack([normalise(e_emb[enrol.label.to_numpy() == n].mean(axis=0)) for n in names])
        sims = q_emb @ protos.T
        order = np.argsort(-sims, axis=1)
        s1, s2 = sims[np.arange(len(sims)), order[:, 0]], sims[np.arange(len(sims)), order[:, 1]]
        correct = np.array(names)[order[:, 0]] == query.label.to_numpy()
        passed, clear = s1 >= cal["threshold"], (s1 - s2) >= cal["margin"]
        out[f"{shots}-shot"] = {
            "enrolment_photos_per_product": shots, "query_pills": int(len(query)), "top1": float(correct.mean()),
            "match_correct": int((passed & clear & correct).sum()), "match_wrong": int((passed & clear & ~correct).sum()),
            "flip": int((passed & ~clear).sum()), "rejected_as_unknown": int((~passed).sum()),
        }
    return out


# ---- C3PI holdout -----------------------------------------------------------------------------------------------------
def eval_c3pi_holdout(model, device, out_dir, cal):
    cache = out_dir / "c3pi_holdout_embeddings.npz"
    manifest = load_manifest(EPILLS_ROOT / "data/processed/c3pi_splits/manifest.csv")
    df = manifest[manifest.split == "epillid_holdout"]
    if cache.exists():
        z = np.load(cache)
        emb, lab = z["emb"], z["lab"]
    else:
        vocab = Vocab(manifest[manifest.split == "train"])
        ds = PillCrops(df, EPILLS_ROOT / "data/interim/c3pi/crops/images", eval_transform(), vocab)
        emb, lab = embed(model, DataLoader(ds, batch_size=128, num_workers=8), device)
        np.savez(cache, emb=emb, lab=lab)
    closed = closed_set_top1(emb, lab, K_ENROL)
    closed.pop("query_index"), closed.pop("correct_per_query")
    return {"protocol": "C3PI everyday photos of the ePillID holdout types (split epillid_holdout); enrolment 3 "
                        "photos per medication; unknown pills from held-out types outside the patient's list",
            "photos": int(len(df)), "medications": int(df.label.nunique()),
            "closed_set_all": closed, "open_set": open_set_report(emb, lab, cal)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sets", nargs="+", default=["c3pi_holdout", "epillid", "mediseg"])
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run
    cal = json.loads((run / "calibration.json").read_text())
    out_dir = run / "eval"
    out_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = HybridPillNet.from_checkpoint(run / "best.pt", device)
    print(f"model {run / 'best.pt'} | threshold {cal['threshold']:.4f} | margin {cal['margin']:.3f}", flush=True)

    for name in args.sets:
        if name == "c3pi_holdout":
            result = eval_c3pi_holdout(model, device, out_dir, cal)
        elif name == "epillid":
            result = eval_epillid(model, device, out_dir)
        elif name == "mediseg":
            result = eval_mediseg(model, device, out_dir, cal)
        else:
            raise SystemExit(f"unknown set {name}")
        result["calibration"] = {"threshold": cal["threshold"], "margin": cal["margin"]}
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(f"\n== {name}\n{json.dumps(result, indent=1)}", flush=True)


if __name__ == "__main__":
    main()
