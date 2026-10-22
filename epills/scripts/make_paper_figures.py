"""
Figures for the course paper (NeurIPS format), from the result files in this repository; nothing is typed in by hand
except the published ePillID reference row and the diagram labels.

Each figure is written as PDF (vector, fonts embedded, for LaTeX) and PNG (300 dpi, for Word) to docs/paper/figures/,
plus figures.json listing each figure's file, caption and source files.

Style: print-ready for the NeurIPS template. Times (the template's font) at 7-8 pt for a 5.5 in text width; no chart
titles (the LaTeX caption names the figure); bold (a)/(b) panel labels; thin axes, light y-grid only; the validated
categorical palette (blue, orange, aqua, yellow) with direct labels.

Usage, from the epills folder:  .venv\\Scripts\\python scripts\\make_paper_figures.py [--only fig03,fig07]
"""
import argparse
import json
import sqlite3
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "docs/paper/figures"
RUNS = ROOT / "outputs/runs"
EV = RUNS / "stage2_run4/eval"
CROPS = ROOT / "data/interim/c3pi/crops/images"
MANIFEST = ROOT / "data/processed/c3pi_splits/manifest.csv"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
INK, INK2, AXIS, GRID, GREY = "#111111", "#555555", "#444444", "#e4e4e4", "#c4c4c4"
SEQ = ["#e8f1fc", "#c3dbf6", "#93bdee", "#5f9be2", "#3279cf", "#1d5aa6", "#103d75"]
FULL, HALF = 5.5, 2.7  # NeurIPS text width and half width, inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.0,
    "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 5, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    "figure.facecolor": "white",
})
REGISTRY = []


def load(path):
    return json.loads(Path(path).read_text())


def manifest():
    m = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    m["pill_id"] = m.ndc9 + "_" + m.part
    return m


def catalogue():
    from epills.core.catalogue.store import Catalogue
    return Catalogue(ROOT / "data/processed/catalogue")


def short_name(p, pid=""):
    name = (p.get("rxnorm_name") or p.get("name") or pid) if p else pid
    for a, b in ((" Oral Tablet", ""), (" Oral Capsule", " capsule"), (" Extended Release", " ER"),
                 ("hydrochloride", "HCl"), (" MG", " mg"), ("24 HR ", "")):
        name = name.replace(a, b)
    return name


def save(fig, name, caption, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    REGISTRY.append({"name": name, "png": f"{name}.png", "pdf": f"{name}.pdf", "caption": caption,
                     "sources": sources})
    print("wrote", name)


def panel(ax, letter, text=""):
    """Bold panel letter and a plain short description, left-aligned above the axes."""
    ax.set_title(rf"$\mathbf{{({letter})}}$ {text}", loc="left")


def hgrid(ax):
    """Horizontal bars: vertical grid lines instead of horizontal ones."""
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)


def bar_labels(ax, bars, fmt="{:.1f}", size=6.5):
    for b in bars:
        h = b.get_height()
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=size, color=INK)


def hbar_labels(ax, bars, fmt="{:.1f}", size=6.5):
    for b in bars:
        w = b.get_width()
        ax.annotate(fmt.format(w), (w, b.get_y() + b.get_height() / 2), xytext=(2, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=size, color=INK)


def stacked_hbar(ax, rows, parts, xmax=100):
    """rows: [(label, {key: share 0-1})]; parts: [(key, legend label, colour)]. Big segments are labelled inside;
    the small ones are listed after the bar, in legend order."""
    for yi, (_, shares) in enumerate(rows):
        left, small = 0, []
        for key, lab, col in parts:
            v = shares[key] * 100
            ax.barh(yi, v, left=left, color=col, height=0.6, edgecolor="white", linewidth=0.8,
                    label=lab if yi == 0 else None)
            if v >= 9:
                ax.text(left + v / 2, yi, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white")
            elif v > 0:
                small.append(f"{v:.2f}" if v < 1 else f"{v:.1f}")
            left += v
        if small:
            ax.text(left + 1.5, yi, " / ".join(small), ha="left", va="center", fontsize=6.2, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]), ax.invert_yaxis()
    ax.set_xlim(0, xmax), ax.set_xticks([0, 25, 50, 75, 100])
    hgrid(ax)
    ax.tick_params(axis="y", length=0)


# ---- diagram helpers ---------------------------------------------------------------------------------------------
LIGHT = {"blue": ("#eef4fc", BLUE), "orange": ("#fdf0e9", ORANGE), "aqua": ("#e8f6f0", AQUA),
         "grey": ("#f5f5f4", "#9a9a96")}


def canvas(w, h):
    """A figure whose data coordinates are inches, so text and boxes size predictably."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w), ax.set_ylim(0, h), ax.set_axis_off()
    return fig, ax


def node(ax, x, y, w, h, title, sub="", tone="blue", title_size=7.3, sub_size=6.2):
    fc, ec = LIGHT[tone]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", fc=fc, ec=ec, lw=0.8))
    if sub:  # title and sub-text centred as one block
        n_title, n_sub = title.count("\n") + 1, sub.count("\n") + 1
        title_h, sub_h, gap = 0.125 * n_title, 0.112 * n_sub, 0.05
        top = y + h / 2 + (title_h + gap + sub_h) / 2
        ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK,
                linespacing=1.1)
        ax.text(x + w / 2, top - title_h - gap, sub, ha="center", va="top", fontsize=sub_size, color=INK2,
                linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, weight="bold", color=INK)
    return (x, y, w, h)


def region(ax, x, y, w, h, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc="none", ec="#b9b9b5",
                                lw=0.7, ls=(0, (3, 2))))
    ax.text(x + 0.07, y + h - 0.05, label, ha="left", va="top", fontsize=6.3, style="italic", color=INK2)


def edge(ax, a, b, label=None, rad=0.0, lpos=0.5, loff=(0, 0.06)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=AXIS, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        x, y = a[0] + (b[0] - a[0]) * lpos + loff[0], a[1] + (b[1] - a[1]) * lpos + loff[1]
        ax.text(x, y, label, ha="center", va="bottom", fontsize=6, color=INK2,
                bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none"})


def thumb(ax, path, x, y, size):
    ax.imshow(Image.open(path).convert("RGB").resize((256, 256)), extent=(x, x + size, y, y + size), zorder=3)
    ax.add_patch(plt.Rectangle((x, y), size, size, fill=False, ec="#9a9a96", lw=0.6, zorder=4))


def example_crop():
    m = manifest()
    return CROPS / m[m.pill_id == "627560143_1"].sort_values("crop_path").crop_path.iloc[1]


# ---- 1. system overview --------------------------------------------------------------------------------------------
def fig01_system():
    W, H = FULL, 2.5
    fig, ax = canvas(W, H)
    d = -0.25  # the top lane, shifted down to close the gap under the bottom lane's boxes
    region(ax, 0.02, 1.27 + d, W - 0.04, 1.46, "Pill verification (this paper's model)")
    region(ax, 0.02, 0.02, W - 0.04, 0.88, "Documents and medication safety")
    thumb(ax, example_crop(), 0.12, 1.62 + d, 0.62)
    ax.text(0.43, 1.52 + d, "dose photo", ha="center", va="top", fontsize=6.2, color=INK2)
    finder = node(ax, 0.92, 1.72 + d, 0.88, 0.48, "Pill finder", "boxes, count,\nquality flags", "grey")
    emb = node(ax, 1.98, 1.72 + d, 0.98, 0.48, "Hybrid CNN–ViT", "256-d unit embedding")
    ocr = node(ax, 1.98, 1.36 + d, 0.98, 0.28, "Imprint OCR", tone="orange", title_size=6.8)
    gal = node(ax, 3.18, 2.0 + d, 1.02, 0.6, "Gallery match", "patient's medicines\nthreshold τ, margin m")
    cat = node(ax, 3.18, 1.36 + d, 1.02, 0.56, "Catalogue identify", "4,936 products\nimprint re-ranking")
    out = node(ax, 4.42, 1.66 + d, 0.98, 0.6, "Answer", "match / unknown /\nretake + confidence", "grey")
    edge(ax, (0.76, 1.96 + d), (finder[0], 1.96 + d), "crop")
    edge(ax, (finder[0] + finder[2], 1.96 + d), (emb[0], 1.96 + d))
    edge(ax, (emb[0] + emb[2], 2.06 + d), (gal[0], 2.3 + d), rad=-0.15)
    edge(ax, (emb[0] + emb[2], 1.86 + d), (cat[0], 1.64 + d), rad=0.15)
    edge(ax, (ocr[0] + ocr[2], 1.53 + d), (gal[0], 2.12 + d), rad=-0.25)
    edge(ax, (ocr[0] + ocr[2], 1.47 + d), (cat[0], 1.47 + d))
    edge(ax, (gal[0] + gal[2], 2.3 + d), (out[0], 2.08 + d), rad=0.15)
    edge(ax, (cat[0] + cat[2], 1.64 + d), (out[0], 1.84 + d), rad=-0.15)
    yb = 0.16  # bottom lane boxes
    label = node(ax, 0.12, yb, 0.98, 0.5, "Pharmacy label", "sticker or box", "grey")
    reader = node(ax, 1.3, yb, 1.1, 0.5, "Label reader", "MOH fields, RxNorm", "orange")
    safety = node(ax, 2.6, yb, 1.6, 0.5, "Safety rules", "DDInter interactions, duplicates", "aqua")
    app = node(ax, 4.42, yb, 0.98, 0.5, "App and agent", "patient confirms", "grey")
    edge(ax, (label[0] + label[2], yb + 0.25), (reader[0], yb + 0.25))
    edge(ax, (reader[0] + reader[2], yb + 0.25), (safety[0], yb + 0.25))
    edge(ax, (safety[0] + safety[2], yb + 0.25), (app[0], yb + 0.25))
    edge(ax, (out[0] + out[2] / 2, out[1]), (app[0] + app[2] / 2, app[1] + app[3]))
    save(fig, "fig01_system_overview",
         "PharmaScan. A dose photo is checked and each pill is cropped and embedded by the hybrid CNN–ViT; the "
         "embedding is matched against the patient's own enrolled medicines (open-set, with calibrated rejection) or, "
         "for an unknown pill, against a 4,936-product catalogue. An imprint reader checks both answers. A label reader "
         "turns the pharmacy sticker into structured fields (Singapore MOH format) and safety rules screen "
         "interactions; the app's agent decides, and the patient confirms.",
         ["serving/app/main.py", "docs/API.md"])


# ---- 2. architecture -----------------------------------------------------------------------------------------------
def fig02_architecture():
    W, H = FULL, 2.3
    fig, ax = canvas(W, H)
    thumb(ax, example_crop(), 0.05, 0.86, 0.62)
    ax.text(0.36, 0.78, "224×224 crop", ha="center", va="top", fontsize=6.2, color=INK2)
    region(ax, 0.82, 0.08, 1.5, 2.16, "pretrained backbones")
    vgg = node(ax, 0.9, 1.3, 1.34, 0.72, "VGG16 + CBAM", "5 conv blocks (ImageNet)\nchannel + spatial attention\nglobal avg. pool, FC 256")
    vit = node(ax, 0.9, 0.2, 1.34, 0.72, "ViT-B/16", "16×16 patches, 12 layers\n(ImageNet-21k)\n[CLS] pooler output")
    region(ax, 2.48, 0.08, 1.7, 2.16, "fusion and refinement")
    cap = node(ax, 2.56, 0.84, 0.74, 0.7, "Capsule\nfusion", "dynamic\nrouting ×2", title_size=7.0, sub_size=6.0)
    ref = node(ax, 3.4, 0.84, 0.7, 0.7, "Trans-\nformer", "4 tokens × 64\nmean pool", title_size=7.0, sub_size=6.0)
    region(ax, 4.32, 0.08, 1.15, 2.16, "heads and losses")
    emb = node(ax, 4.4, 1.46, 1.0, 0.58, "Embedding", "256-d, unit length\nArcFace s=32, m=0.3", "orange")
    shp = node(ax, 4.4, 0.82, 1.0, 0.5, "Shape", "softmax, CE ×0.2", "aqua")
    col = node(ax, 4.4, 0.2, 1.0, 0.5, "Colour", "sigmoid, BCE ×0.2", "aqua")
    edge(ax, (0.69, 1.32), (vgg[0], 1.66), rad=-0.2)
    edge(ax, (0.69, 1.02), (vit[0], 0.56), rad=0.2)
    edge(ax, (vgg[0] + vgg[2], 1.66), (cap[0], 1.36), "256", rad=-0.15, lpos=0.45)
    edge(ax, (vit[0] + vit[2], 0.56), (cap[0], 1.0), "768", rad=0.15, lpos=0.45, loff=(0, -0.12))
    edge(ax, (cap[0] + cap[2], 1.19), (ref[0], 1.19), "256", loff=(0, 0.03))
    edge(ax, (ref[0] + ref[2], 1.36), (emb[0], 1.75), "64", rad=-0.15, lpos=0.5)
    edge(ax, (ref[0] + ref[2], 1.19), (shp[0], 1.07))
    edge(ax, (ref[0] + ref[2], 1.0), (col[0], 0.45), rad=0.15)
    save(fig, "fig02_architecture",
         "The hybrid network. VGG16 with CBAM attention and ViT-B/16 see the same crop; a capsule block fuses their "
         "features by dynamic routing and a transformer block refines the result into a unit-length 256-d embedding "
         "trained with ArcFace, with auxiliary shape and colour heads. Numbers on the arrows are feature sizes.",
         ["src/epills/core/models/hybrid.py", "configs/stage2_run4.yaml"])


# ---- 3-4. data -----------------------------------------------------------------------------------------------------
def fig03_data():
    m = manifest()
    per = m.groupby("pill_id").size()
    names = {"train": "Train", "val_enrolled": "Val. (enrolled)", "val_unknown": "Val. (unknown)",
             "epillid_holdout": "Test (held-out)"}
    splits = m.groupby("split").agg(photos=("crop_path", "size"), groups=("label", "nunique")).reindex(list(names))
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 1.9), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(wspace=0.7)
    a.hist(per.clip(upper=60), bins=np.arange(0, 62, 2), color=BLUE, edgecolor="white", linewidth=0.4)
    panel(a, "a", "Photos per medicine")
    a.set_xlabel("everyday photos per medicine (≥ 60 in last bin)"), a.set_ylabel("medicines")
    a.annotate(f"{len(per):,} medicines\nmedian {per.median():.0f} photos", (0.97, 0.95), xycoords="axes fraction",
               ha="right", va="top", fontsize=6.5, color=INK2)
    y = np.arange(len(names))
    bars = b.barh(y, splits.photos, color=BLUE, height=0.55)
    b.set_yticks(y, [f"{names[s]}\n{int(splits.groups[s]):,} groups" for s in names]), b.invert_yaxis()
    hbar_labels(b, bars, "{:,.0f}")
    panel(b, "b", "Photos per split")
    b.set_xlabel("photos"), b.set_xlim(0, splits.photos.max() * 1.3), hgrid(b), b.tick_params(axis="y", length=0)
    save(fig, "fig03_data_statistics",
         "C3PI everyday photos. (a) Photos per medicine. (b) Splits are made by appearance group (look-alike products "
         "share a group), so no medicine or look-alike appears in two splits; the test split holds the medicines of "
         "ePillID's official holdout fold.",
         ["data/processed/c3pi_splits/manifest.csv"])


def fig04_examples():
    m = manifest()
    cat = catalogue()
    picks = ["627560143_1", "001723758_1", "000691540_1", "683820052_1"]  # validation medicines
    fig, axes = plt.subplots(len(picks), 5, figsize=(FULL, 4.4))
    for r, pid in enumerate(picks):
        rows = m[m.pill_id == pid].sort_values("crop_path").head(4)
        p = cat.product(pid)
        ref = cat.image_file(pid, p["images"][0]["n"]) if p and p["images"] else None
        tiles = [("catalogue reference", Image.open(ref) if ref else None)] + \
                [(f"everyday photo {k + 1}", Image.open(CROPS / c)) for k, c in enumerate(rows.crop_path)]
        for c, (label, img) in enumerate(tiles):
            ax = axes[r, c]
            ax.set_axis_off()
            if img is not None:
                ax.imshow(img)
            if r == 0:
                ax.set_title(label, fontsize=7, loc="center", color=INK2)
        axes[r, 0].text(-0.08, 0.5, short_name(p, pid)[:34] + f"\nimprint {p['imprint'] or '–'}",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=7, color=INK)
    fig.subplots_adjust(left=0.27, wspace=0.04, hspace=0.06)
    save(fig, "fig04_qualitative_examples",
         "Qualitative examples (C3PI validation medicines, never trained on): the catalogue reference picture and "
         "four everyday crops per medicine. Lighting, background, white balance, side shown and rotation vary; "
         "debossed imprints are often hard to read.",
         ["data/interim/c3pi/crops/images", "data/processed/catalogue"])


# ---- 5. training ---------------------------------------------------------------------------------------------------
def fig05_training():
    runs = [("stage2_baseline", "Run 1 (new layers at $10^{-3}$)"), ("stage2_run2", "Run 2 (lower rates)"),
            ("stage2_run3", "Run 3 (+ reference photos)"), ("stage2_run4", "Run 4 (+ augmentation)")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.1))
    fig.subplots_adjust(wspace=0.3)
    best = []
    for (run, label), col in zip(runs, SERIES):
        h = pd.read_csv(RUNS / run / "history.csv")
        x = np.arange(1, len(h) + 1)
        a.plot(x, h.top1_gallery10 * 100, color=col, label=label)
        best.append(h.top1_gallery10.max() * 100)
        if "embedding_dims" in h:
            b.plot(x, h.embedding_dims, color=col, label=label)
    a.annotate(f"best {best[0]:.1f}", (60, best[0]), xytext=(0, 5), textcoords="offset points", ha="right",
               fontsize=6.5, color=INK2)
    a.annotate(f"best {min(best[1:]):.1f}–{max(best[1:]):.1f}", (60, max(best[1:])), xytext=(0, -10),
               textcoords="offset points", ha="right", fontsize=6.5, color=INK2)
    b.axhline(3, color=BLUE, lw=1.2, ls=(0, (3, 2)))
    b.annotate("run 1: ≈ 3 (collapsed)", (59, 3.5), ha="right", fontsize=6.5, color=INK2)
    p1 = len(pd.read_csv(RUNS / "stage2_run4/history.csv").query("phase == 1"))
    for ax in (a, b):
        ax.axvline(p1 + 0.5, color=GREY, lw=0.7, ls=":")
        ax.set_xlabel(f"epoch (dotted: phase 2 from epoch {p1 + 1})")
    panel(a, "a", "Validation top-1 (10 medicines)"), a.set_ylabel("top-1 (%)"), a.set_ylim(30, 102)
    panel(b, "b", "Effective embedding dimensions"), b.set_ylabel("effective dimensions"), b.set_ylim(0, 18)
    a.legend(loc="lower right", fontsize=6.3)
    save(fig, "fig05_training_curves",
         "Training. (a) Validation top-1 within a gallery of 10 medicines never trained on, each enrolled from 3 photos. "
         "(b) Effective dimensionality of the embedding (exp of the entropy of the variance spectrum). With the new "
         "layers trained at 1e-3 the embedding collapsed to about 3 dimensions; lower rates, a separate rate for the "
         "ArcFace table and a margin warm-up fixed it.",
         [f"outputs/runs/{r}/history.csv" for r, _ in runs])


# ---- 6. calibration ------------------------------------------------------------------------------------------------
def fig06_calibration():
    t = pd.read_csv(RUNS / "stage2_run4/calibration_table.csv").query("gallery_size == 10 and k_enrol == 3")
    cal = load(RUNS / "stage2_run4/calibration.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.0))
    fig.subplots_adjust(wspace=0.3)
    a.plot(t.margin, t.correct_match * 100, color=BLUE, label="named correctly")
    a.plot(t.margin, t.own_flip * 100, color=ORANGE, label="asked to retake")
    a.plot(t.margin, t.wrong_match * 100, color=AQUA, label="named wrongly")
    a.axvline(cal["margin"], color=GREY, lw=0.7, ls=":")
    a.annotate(f"$m$ = {cal['margin']}", (cal["margin"], 22), xytext=(3, 0), textcoords="offset points", fontsize=6.5,
               color=INK2)
    panel(a, "a", "Margin"), a.set_xlabel("margin $m$"), a.set_ylabel("own pills (%)")
    a.legend(loc="center left", fontsize=6.3, bbox_to_anchor=(0.0, 0.52))
    w = pd.DataFrame(cal["window_sweep"])
    b.plot(w.window, w.correct_match * 100, color=BLUE, marker="o")
    for x, y in zip(w.window, w.correct_match * 100):
        b.annotate(f"{y:.1f}", (x, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6.3)
    b.axvline(cal["window"], color=GREY, lw=0.7, ls=":")
    b.set_ylim(w.correct_match.min() * 100 - 0.3, w.correct_match.max() * 100 + 0.35)
    panel(b, "b", "Crop window"), b.set_xlabel("window (× pill box long side)"), b.set_ylabel("named correctly (%)")
    save(fig, "fig06_calibration",
         "Settings chosen on validation (1,000 simulated patients, 10 medicines, 3 enrolment photos; τ rejects 95% of "
         "other medicines). (a) The margin trades correct names for retakes while wrong names stay near zero. (b) The "
         "crop window (note the narrow y-range).",
         ["outputs/runs/stage2_run4/calibration_table.csv", "outputs/runs/stage2_run4/calibration.json"])


# ---- 7-10. results -------------------------------------------------------------------------------------------------
def fig07_results():
    runs = [("stage2_run2", "Run 2"), ("stage2_run3", "Run 3"), ("stage2_run4", "Run 4 (served)")]
    metrics = [("C3PI held-out\ntop-1, 200 medicines", lambda r: load(RUNS / r / "eval/c3pi_holdout.json")
                ["closed_set_all"]["top1"]),
               ("MEDISEG (other phones)\ntop-1, 32 products", lambda r: load(RUNS / r / "eval/mediseg.json")
                ["closed_set_32"]["top1"]),
               ("ePillID (strict)\nMAP@1, two-sided", lambda r: load(RUNS / r / "eval/epillid.json")
                ["two_sided"]["MAP@1"])]
    fig, ax = plt.subplots(figsize=(FULL, 2.0))
    x = np.arange(len(metrics))
    wdt = 0.25
    for k, ((run, label), col) in enumerate(zip(runs, SERIES)):
        bars = ax.bar(x + (k - 1) * wdt, [f(run) * 100 for _, f in metrics], wdt - 0.03, color=col, label=label)
        bar_labels(ax, bars)
    ax.set_xticks(x, [n for n, _ in metrics]), ax.set_ylabel("%"), ax.set_ylim(0, 100)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=3, loc="upper right")
    save(fig, "fig07_main_results",
         "Test results on medicines never seen in training, each scored once. Run 3 adds C3PI studio reference photos; "
         "run 4 adds augmentation for other phones and jittered crops.",
         [f"outputs/runs/{r}/eval/{f}.json" for r, _ in runs for f in ("c3pi_holdout", "mediseg", "epillid")])


def fig08_open_set():
    c = load(EV / "c3pi_holdout.json")["open_set"]["gallery_10"]["at_calibration"]
    m = load(EV / "mediseg.json")["open_set"]["gallery_10"]["at_calibration"]
    rows = [("C3PI held-out", c), ("MEDISEG", m)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 1.55), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(wspace=0.08)
    stacked_hbar(a, rows, [("correct_match", "named correctly", BLUE), ("own_flip", "retake", AQUA),
                           ("own_rejected", "called unknown", YELLOW), ("wrong_match", "named wrongly", ORANGE)],
                 xmax=114)
    stacked_hbar(b, rows, [("unknown_rejected", "rejected", BLUE), ("unknown_flip", "retake", AQUA),
                           ("unknown_matched", "accepted as theirs", ORANGE)], xmax=114)
    b.set_yticklabels([])
    panel(a, "a", "The patient's own pills"), panel(b, "b", "Pills that are not theirs")
    for ax in (a, b):
        ax.set_xlabel("%")
        ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.42), fontsize=6.3)
    save(fig, "fig08_open_set_outcomes",
         "What the patient sees (run 4, embedding only; 10 medicines, 3 enrolment photos, calibrated τ and m). Small "
         "shares are listed after each bar in legend order. In C3PI conditions most own pills are named correctly and "
         "wrong names are rare; on other phones the threshold transfers badly and most own pills are called unknown, "
         "which is safe but unhelpful.",
         ["outputs/runs/stage2_run4/eval/c3pi_holdout.json", "outputs/runs/stage2_run4/eval/mediseg.json"])


def fig09_robustness():
    runs = [("stage2_run2", "Run 2"), ("stage2_run3", "Run 3"), ("stage2_run4", "Run 4")]
    sizes = ["under 50 px", "50-100 px", "over 100 px"]
    shots = ["1-shot", "5-shot", "10-shot"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.0))
    fig.subplots_adjust(wspace=0.3)
    x = np.arange(len(sizes))
    for k, ((run, label), col) in enumerate(zip(runs, SERIES)):
        d = load(RUNS / run / "eval/mediseg.json")
        bars = a.bar(x + (k - 1) * 0.27, [d["closed_set_32"]["by_pill_size"][s]["top1"] * 100 for s in sizes],
                     0.25, color=col, label=label)
        bar_labels(a, bars, "{:.0f}", size=6.2)
        ys = [d["deploy"][s]["top1"] * 100 for s in shots]
        b.plot([1, 5, 10], ys, color=col, marker="o", label=label)
        b.annotate(f"{ys[-1]:.0f}", (10, ys[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                   fontsize=6.3, color=INK2)
    a.set_xticks(x, ["< 50 px", "50–100 px", "> 100 px"]), a.set_ylim(0, 80), a.tick_params(axis="x", length=0)
    panel(a, "a", "By pill size in the photo"), a.set_ylabel("MEDISEG top-1 (%)"), a.set_xlabel("pill long side")
    a.legend(ncol=3, fontsize=6.3, loc="upper left")
    b.set_xticks([1, 5, 10]), b.set_xlim(0, 11.5)
    panel(b, "b", "By enrolment photos (deploy split)")
    b.set_xlabel("enrolment photos per product"), b.set_ylabel("MEDISEG top-1 (%)"), b.legend(fontsize=6.3)
    q = load(EV / "mediseg.json")["deploy"]["1-shot"].get("query_pills")
    save(fig, "fig09_robustness",
         "Robustness on MEDISEG (32 Hong Kong products, other phones and cluttered scenes, none in training). (a) Small "
         f"pills are the hard case. (b) On MEDISEG's deploy split ({q} query pills) more enrolment photos help only "
         "partly: the gap is mostly the photo domain.",
         [f"outputs/runs/{r}/eval/mediseg.json" for r, _ in runs])


def fig10_epillid():
    rows = [("Usuyama et al. (best)", 0.9201, 0.9119)]
    for run, label in (("stage2_run5_official", "Run 5"), ("stage2_run6_official", "Run 6"),
                       ("stage2_run7_official", "Run 7")):
        e, i = load(RUNS / run / "eval/epillid.json")["two_sided"], load(RUNS / run / "eval/epillid_imprint.json")
        rows.append((f"{label}", e["MAP@1"], e["GAP@1"]))
        rows.append((f"{label} + imprint", i["two_sided"]["imprint"]["MAP@1"], i["two_sided"]["imprint"]["GAP@1"]))
    fig, ax = plt.subplots(figsize=(FULL, 2.4))
    y = np.arange(len(rows))
    b1 = ax.barh(y - 0.19, [r[1] * 100 for r in rows], 0.36, color=BLUE, label="MAP@1")
    b2 = ax.barh(y + 0.19, [r[2] * 100 for r in rows], 0.36, color=ORANGE, label="GAP@1")
    hbar_labels(ax, b1), hbar_labels(ax, b2)
    ax.set_yticks(y, [r[0] for r in rows]), ax.invert_yaxis(), ax.set_xlim(0, 100), hgrid(ax)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("%"), ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig10_epillid_benchmark",
         "The published ePillID benchmark (official protocol, fold 4, two-sided queries against references of all "
         "4,902 types). Runs 5–7 are trained on ePillID's own training folds. Imprint re-ranking uses OCR information "
         "the published model did not use.",
         [f"outputs/runs/{r}/eval/epillid*.json" for r in ("stage2_run5_official", "stage2_run6_official",
                                                            "stage2_run7_official")])


# ---- 11-13. imprint and identify -----------------------------------------------------------------------------------
def fig11_imprint():
    from epills.core.imprint.match import ImprintIndex, read_hits
    m = manifest()
    meta = pd.read_csv(ROOT / "data/interim/c3pi/metadata.csv", dtype=str, keep_default_na=False,
                       usecols=["path", "rating_imprint", "imprint_type"])
    val = m[m.split.isin(["val_enrolled", "val_unknown"]) & (m.imprint != "")].merge(meta, on="path", how="left")
    reads = load(ROOT / "data/interim/ocr/large_c92/validation_all.json")
    con = sqlite3.connect(ROOT / "data/processed/catalogue/catalogue.sqlite")
    w = ImprintIndex([r[0] or "" for r in con.execute("select imprint from products")]).weight
    val["read"] = [bool(read_hits(reads.get(str(CROPS / p), []), {0: i}, w, 0.5))
                   for p, i in zip(val.crop_path, val.imprint)]
    size = pd.to_numeric(val.size_mm, errors="coerce")
    val["size_band"] = pd.cut(size, [0, 7, 10, 14, 99], labels=["< 7 mm", "7–10 mm", "10–14 mm", "> 14 mm"])
    groups = [("imprint_type", "a", "Imprint type", None),
              ("rating_imprint", "b", "Rater's judgement", ["Clear", "Partial", "Can not Identify"]),
              ("size_band", "c", "Pill size", ["< 7 mm", "7–10 mm", "10–14 mm", "> 14 mm"]),
              ("shape", "d", "Shape", ["ROUND", "OVAL", "CAPSULE"])]
    fig, axes = plt.subplots(1, 4, figsize=(FULL, 1.85), sharey=True, gridspec_kw={"width_ratios": [2.4, 3, 4, 3]})
    fig.subplots_adjust(wspace=0.3)
    for ax, (col, letter, title, order) in zip(axes, groups):
        g = val.groupby(col, observed=True)["read"].agg(["mean", "size"])
        g = g.reindex(order) if order else g[g["size"] > 50]
        bars = ax.bar(range(len(g)), g["mean"] * 100, color=BLUE, width=0.6)
        bar_labels(ax, bars, "{:.0f}", size=6.3)
        labels = [str(i).replace("Can not Identify", "unreadable").capitalize().replace("Mm", "mm") for i in g.index]
        ax.set_xticks(range(len(g)), labels, rotation=35, ha="right", fontsize=6.6)
        ax.tick_params(axis="x", length=0)
        panel(ax, letter, title), ax.set_ylim(0, 85)
    axes[0].set_ylabel("own imprint read (%)")
    save(fig, "fig11_imprint_reading",
         f"Imprint reading (Florence-2-large, zero-shot, {len(val):,} validation photos of medicines with an imprint): "
         "the share of photos on which a token of the pill's own imprint is read, by imprint type, C3PI rater's "
         "judgement of visibility, pill size and shape.",
         ["data/interim/ocr/large_c92/validation_all.json", "data/interim/c3pi/metadata.csv"])


def fig12_identify():
    t = load(EV / "identify_topk_large_c92.json")
    r = load(EV / "identify_rerank_large_c92.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.25), gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(wspace=0.42)
    ks = [1, 3, 5, 10, 20]
    series = [("look only", t["look"], BLUE), ("+ imprint re-ranking", t["imprint"], ORANGE),
              ("+ imprint, own imprint read", t["imprint_by_own_read"]["True"], AQUA)]
    for label, d, col in series:
        ys = [d[f"top{k}"] * 100 for k in ks]
        a.plot(ks, ys, color=col, marker="o", label=label)
        a.annotate(f"{ys[0]:.1f}", (1, ys[0]), xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                   fontsize=6.3, color=INK2)
    a.set_xticks(ks), a.set_xlim(-2.5, 21), a.set_ylim(60, 101), a.set_xlabel("$k$")
    a.set_ylabel("right product in top $k$ (%)")
    panel(a, "a", "Held-out medicines"), a.legend(loc="lower right", fontsize=6.3)
    g = pd.DataFrame(r["validation_grid"]).pivot(index="alpha", columns="top_k", values="top1") * 100
    lo, hi = np.sort(g.values.ravel())[g.shape[1]], g.values.max()  # skip the alpha = 0 row in the colour range
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    b.imshow(g.values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.values[i, j]
            b.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.8,
                   color="white" if v > (lo + hi) / 2 else INK)
    b.set_xticks(range(g.shape[1]), g.columns), b.set_yticks(range(g.shape[0]), g.index), b.grid(False)
    b.tick_params(length=0)
    for s in ("left", "bottom"):
        b.spines[s].set_visible(False)
    b.set_xlabel("products re-ranked $K$"), b.set_ylabel(r"imprint weight $\alpha$")
    panel(b, "b", "Validation top-1 (%)")
    ch = r["chosen"]
    b.add_patch(plt.Rectangle((list(g.columns).index(ch["top_k"]) - 0.5, list(g.index).index(ch["alpha"]) - 0.5),
                              1, 1, fill=False, ec=ORANGE, lw=1.4))
    save(fig, "fig12_identify_topk",
         "Identifying an unknown pill against the 4,936-product catalogue. (a) Top-k on held-out medicines: imprint "
         "re-ranking helps most when the pill's own imprint is read. (b) The re-ranking weight α and depth K on "
         "validation; the box marks the served setting, and α = 0 is the look alone.",
         ["outputs/runs/stage2_run4/eval/identify_topk_large_c92.json",
          "outputs/runs/stage2_run4/eval/identify_rerank_large_c92.json"])


def fig13_confidence():
    d = load(EV / "identify_confidence_large_c92.json")["test"]
    rel = pd.DataFrame(d["reliability"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL, 2.0), gridspec_kw={"width_ratios": [1, 1.35]})
    fig.subplots_adjust(wspace=0.6)
    a.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls=":")
    a.plot(rel.stated, rel.observed, color=BLUE, marker="o")
    for _, row in rel.iterrows():
        a.annotate(f"n={row.pills}", (row.stated, row.observed), xytext=(-4, 3), textcoords="offset points",
                   ha="right", fontsize=6.2, color=INK2)
    a.set_xlim(0, 1), a.set_ylim(0, 1), a.set_xlabel("stated confidence"), a.set_ylabel("observed accuracy")
    a.grid(axis="x", visible=True)
    panel(a, "a", "Reliability")
    verdicts = [("possible_match", "possible match", BLUE), ("uncertain", "uncertain", AQUA),
                ("no_confident_match", "no confident match", YELLOW)]
    rows = [("Catalogue pills\n(C3PI held-out)", {k: d["catalogue_pills_by_verdict"][k]["share"]
                                                  for k, _, _ in verdicts}),
            ("Not in catalogue\n(MEDISEG)", d["other_pills_by_verdict"])]
    stacked_hbar(b, rows, verdicts, xmax=114)
    b.set_xlabel("%"), panel(b, "b", "Verdicts")
    b.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.42, -0.3), fontsize=6.3)
    save(fig, "fig13_identify_confidence",
         "Calibrated confidence for identification. Raw cosine similarities are not probabilities (pills that are not "
         "in the catalogue still score about 0.93), so a logistic model of the top-1 similarity, its gap to the "
         "runner-up and the imprint evidence is fitted on validation. (a) Stated confidence against observed accuracy "
         "on held-out medicines. (b) Verdicts for catalogue pills and for pills not in the catalogue.",
         ["outputs/runs/stage2_run4/eval/identify_confidence_large_c92.json"])


# ---- 14-15. embedding and similarity -------------------------------------------------------------------------------
def holdout():
    z = np.load(EV / "c3pi_holdout_embeddings.npz")
    emb = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True)
    return emb, z["lab"]


def fig14_similarity():
    emb, lab = holdout()
    rng = np.random.default_rng(0)
    protos, held = {}, []
    for l in np.unique(lab):
        idx = np.flatnonzero(lab == l)
        if len(idx) < 4:
            continue
        e = rng.choice(idx, 3, replace=False)
        p = emb[e].mean(0)
        protos[l] = p / np.linalg.norm(p)
        held += [(i, l) for i in np.setdiff1d(idx, e)]
    keys = np.array(list(protos))
    P = np.stack([protos[k] for k in keys])
    own, other = [], []
    for i, l in held:  # a gallery of 10: the photo's own medicine and 9 others drawn at random
        s = P @ emb[i]
        own.append(s[keys == l][0])
        other.append(s[rng.choice(np.flatnonzero(keys != l), 9, replace=False)].max())
    own, other = np.array(own), np.array(other)
    thr = load(RUNS / "stage2_run4/calibration.json")["threshold"]
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    bins = np.linspace(0.4, 1.0, 61)
    ax.hist(own, bins=bins, color=BLUE, alpha=0.25, lw=0)
    ax.hist(own, bins=bins, color=BLUE, histtype="step", lw=1.1,
            label=f"own medicine ({(own >= thr).mean() * 100:.0f}% ≥ τ)")
    ax.hist(other, bins=bins, color=ORANGE, alpha=0.2, lw=0)
    ax.hist(other, bins=bins, color=ORANGE, histtype="step", lw=1.1,
            label=f"closest of 9 others ({(other >= thr).mean() * 100:.0f}% ≥ τ)")
    ax.axvline(thr, color=AXIS, lw=0.7, ls=":")
    ax.annotate(f"τ = {thr:.3f}", (thr, ax.get_ylim()[1] * 0.97), xytext=(-3, 0), textcoords="offset points",
                ha="right", va="top", fontsize=6.5, color=INK2)
    ax.set_xlabel("cosine similarity to a 3-photo prototype"), ax.set_ylabel("photos")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.86), fontsize=6.2)
    save(fig, "fig14_similarity_distributions",
         "Held-out photos (200 new medicines): similarity to their own medicine's prototype (mean of 3 other photos) "
         "and to the closest of 9 other medicines, as in a 10-medicine gallery, with the threshold τ chosen on "
         "validation.",
         ["outputs/runs/stage2_run4/eval/c3pi_holdout_embeddings.npz", "outputs/runs/stage2_run4/calibration.json"])


def fig15_tsne():
    from sklearn.manifold import TSNE
    emb, lab = holdout()
    rows = manifest().query("split == 'epillid_holdout'").reset_index(drop=True)
    assert len(rows) == len(lab)
    cat = catalogue()
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb)
    first = {l: rows.pill_id[np.flatnonzero(lab == l)[0]] for l in np.unique(lab)}
    show = [l for l in pd.Series(lab).value_counts().index  # skip kits, whose names are long
            if not short_name(cat.product(first[l]), first[l]).startswith("{")][:3]
    fig, ax = plt.subplots(figsize=(HALF, 2.3))
    rest = ~np.isin(lab, show)
    ax.scatter(xy[rest, 0], xy[rest, 1], s=1.2, color=GREY, linewidths=0, label="197 other medicines", rasterized=True)
    for l, col in zip(show, SERIES):
        k = lab == l
        pid = first[l]
        ax.scatter(xy[k, 0], xy[k, 1], s=8, color=col, edgecolors="white", linewidths=0.3,
                   label=f"{short_name(cat.product(pid), pid)[:28]} ({k.sum()})")
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=6.2, markerscale=1.4)
    save(fig, "fig15_embedding_tsne",
         "t-SNE of the served model's embeddings for the 4,005 photos of the 200 held-out medicines (never seen in "
         "training). Photos of one medicine form tight clusters; the three single-tablet medicines with the most photos "
         "are highlighted (photo counts in brackets).",
         ["outputs/runs/stage2_run4/eval/c3pi_holdout_embeddings.npz"])


# ---- 16-17. error analysis -----------------------------------------------------------------------------------------
def fig16_lookalikes():
    s = load(EV / "lookalike_strengths_large_c92_distinct.json")
    m = load(EV / "same_maker_imprints_large_c92_distinct.json")
    groups = [(f"Same ingredient, other strength ({s['pairs']:,} pairs)", s["all"]),
              (" – of which same shape and colour", s["look_alike"]),
              (f"Same maker, shared imprint part ({m['pairs']:,} pairs)", m["all"]),
              (" – of which sharing Teva's 'TV'", m["shared 'TV'"])]
    key = lambda d: next(k for k in d["look only"] if k.startswith("named as the"))  # noqa: E731
    fig, (a, b) = plt.subplots(2, 1, figsize=(FULL, 3.3), gridspec_kw={"height_ratios": [1, 1.05]})
    fig.subplots_adjust(hspace=0.65)
    y = np.arange(len(groups))
    for k, (check, col, lab) in enumerate((("look only", BLUE, "look only"),
                                           ("with imprint", ORANGE, "+ imprint check"))):
        bars = a.barh(y + (k - 0.5) * 0.38, [g[check][key(g)] for _, g in groups], 0.36, color=col, label=lab)
        hbar_labels(a, bars, "{:.2f}")
    a.set_yticks(y, [g for g, _ in groups], fontsize=6.8), a.invert_yaxis(), hgrid(a), a.tick_params(axis="y", length=0)
    a.set_xlabel("named as the other medicine (%)"), a.legend(fontsize=6.3, loc="lower right")
    panel(a, "a", "Both medicines enrolled in one gallery")
    worst = m["worst_pairs"][:5]
    yy = np.arange(len(worst))
    bars = b.barh(yy, [w["named_as_the_other_%"] for w in worst], color=BLUE, height=0.55)
    hbar_labels(b, bars, "{:.1f}")

    def pair_name(p):
        name, imprint = p.rsplit(" (", 1)
        name = short_name({"name": name}).replace(" Tablet for Oral Suspension", "").replace(" ER", "")
        return f"{name} ({imprint}"
    b.set_yticks(yy, [" vs ".join(pair_name(p) for p in w["pair"]) for w in worst], fontsize=6.8)
    b.invert_yaxis(), hgrid(b), b.tick_params(axis="y", length=0), b.set_xlabel("named as the other medicine (%)")
    panel(b, "b", "Worst same-maker pairs, with the imprint check (imprints in brackets)")
    b.set_xlim(0, max(w["named_as_the_other_%"] for w in worst) * 1.2)
    sa, sw = s["all"]["look only"], s["all"]["with imprint"]
    save(fig, "fig16_lookalike_error_analysis",
         "Look-alikes, with both medicines enrolled in one gallery (C3PI validation and held-out photos). (a) Wrong "
         "names stay rare; different strengths that look the same are the hardest. The imprint check settles retakes "
         f"(same-ingredient pairs: retakes {sa['ambiguous']:.1f}% → {sw['ambiguous']:.1f}%, named correctly "
         f"{sa['named correctly']:.1f}% → {sw['named correctly']:.1f}%) at the cost of slightly more wrong names. (b) "
         "The worst same-manufacturer pairs share Teva's 'TV' prefix.",
         ["outputs/runs/stage2_run4/eval/lookalike_strengths_large_c92_distinct.json",
          "outputs/runs/stage2_run4/eval/same_maker_imprints_large_c92_distinct.json"])


def fig17_finder():
    p = load(ROOT / "outputs/eval/pill_finder.json")["by_pills"]
    rows = [(f"{n} pill{'s' if n != '1' else ''} ({p[n]['photos']:,})",
             {k: p[n][k] for k in ("exact_count", "under_count", "over_count")}) for n in "123"]
    fig, ax = plt.subplots(figsize=(HALF, 1.45))
    stacked_hbar(ax, rows, [("exact_count", "count right", BLUE), ("under_count", "too few", AQUA),
                            ("over_count", "too many", ORANGE)], xmax=114)
    ax.set_xlabel("photos (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.38), fontsize=6.2)
    save(fig, "fig17_pill_finder",
         "The pill finder on C3PI photos with known pill counts (2- and 3-pill photos composed from single-pill "
         "photos). An earlier finder found exactly one box on only 28% of single-pill photos.",
         ["outputs/eval/pill_finder.json"])


FIGURES = [fig01_system, fig02_architecture, fig03_data, fig04_examples, fig05_training, fig06_calibration,
           fig07_results, fig08_open_set, fig09_robustness, fig10_epillid, fig11_imprint, fig12_identify,
           fig13_confidence, fig14_similarity, fig15_tsne, fig16_lookalikes, fig17_finder]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="comma-separated prefixes, e.g. fig03,fig07")
    args = parser.parse_args()
    only = [o for o in args.only.split(",") if o]
    failed = []
    for f in FIGURES:
        if only and not any(f.__name__.startswith(o) for o in only):
            continue
        try:
            f()
        except Exception:
            failed.append(f.__name__)
            traceback.print_exc()
    reg = OUT / "figures.json"
    old = {r["name"]: r for r in (json.loads(reg.read_text()) if reg.exists() else [])}
    old.update({r["name"]: r for r in REGISTRY})
    reg.write_text(json.dumps(sorted(old.values(), key=lambda r: r["name"]), indent=1), encoding="utf-8")
    print(f"{len(REGISTRY)} figures written to {OUT}; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
