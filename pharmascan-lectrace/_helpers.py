"""
Plumbing for 01_pharmascan.py (lectrace never traces or shows files starting with '_'): loading the exported data and
the textbook layers (dense, batch norm, layer norm, softmax). The steps worth watching live in the lecture file.
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


class Weights:
    """The served model's layers after the backbones (data/weights.npz); shown as a summary, not 645k numbers."""

    def __init__(self, path):
        self.w = dict(np.load(path))

    def __getitem__(self, key):
        return self.w[key]

    def __lectrace__(self):
        return {"layers": "capsule fusion, transformer refinement, embedding head",
                "numbers": int(sum(v.size for v in self.w.values()))}


class Catalogue:
    """Reference embeddings of the 300 catalogue products nearest the foreign pill (of 4,936), with their imprints."""

    def __init__(self, path, products):
        z = np.load(path)
        self.emb, self.owner, self.products = z["emb"], z["owner"], products

    def look(self, query):
        """Each product's look score: its best reference photo's cosine similarity (as the served API)."""
        sims = self.emb @ query
        out = np.full(len(self.products), -1.0)
        np.maximum.at(out, self.owner, sims)
        return out

    def __len__(self):
        return len(self.products)

    def __getitem__(self, i):
        return self.products[i]

    def __lectrace__(self):
        return {"products": len(self.products), "reference photos": int(len(self.emb))}


def load():
    demo = json.loads((HERE / "data/demo.json").read_text(encoding="utf-8"))
    back = dict(np.load(HERE / "data/backbones.npz"))
    return demo, Weights(HERE / "data/weights.npz"), back, Catalogue(HERE / "data/catalogue.npz",
                                                                       demo["catalogue_products"])


# ---- textbook layers (inference mode: dropout off, batch norm uses its running statistics) ----------------------------
def dense(w, name, x):
    return x @ w[f"{name}.W"].T + w[f"{name}.b"]


def relu(x):
    return np.maximum(x, 0.0)


def batch_norm(w, name, x):
    return (x - w[f"{name}.mean"]) / np.sqrt(w[f"{name}.var"] + w[f"{name}.eps"]) * w[f"{name}.gamma"] + w[f"{name}.beta"]


def layer_norm(w, name, x):
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + w[f"{name}.eps"]) * w[f"{name}.g"] + w[f"{name}.b"]


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def squash(s, eps=1e-7):
    """Capsule non-linearity: keeps the direction, maps the length into [0, 1)."""
    n2 = (s ** 2).sum()
    return n2 / (1 + n2) * s / np.sqrt(n2 + eps)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def encoder_layer(w, i, x):
    """One post-norm transformer encoder layer: 4 heads of 64 over the 4 tokens. Returns (tokens, attention)."""
    T = x.shape[0]
    split = lambda t: t.reshape(T, 4, 64).transpose(1, 0, 2)  # noqa: E731  (heads, tokens, 64)
    q, k, v = (split(dense(w, f"refine.{i}.{n}", x)) for n in ("q", "k", "v"))
    attn = softmax(q @ k.transpose(0, 2, 1) / 8.0)  # (heads, tokens, tokens); sqrt(64) = 8
    mixed = dense(w, f"refine.{i}.out", (attn @ v).transpose(1, 0, 2).reshape(T, 256))
    x = layer_norm(w, f"refine.{i}.norm1", x + mixed)
    ffn = dense(w, f"refine.{i}.ffn1", relu(dense(w, f"refine.{i}.ffn0", x)))
    return layer_norm(w, f"refine.{i}.norm2", x + ffn), attn


def compact(text):
    """OCR text -> letters and digits only, upper case ('ECS\\n' -> 'ECS')."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def imprint_scores(texts, catalogue, idx):
    """Per product in idx: the weight (rarity, 0-1) of its rarest recorded imprint token found in the OCR reads."""
    return np.array([max((wt for tok, wt in catalogue[j]["tokens"].items() if any(tok in t for t in texts)),
                         default=0.0) for j in idx])


def similarity_chart(rows, threshold):
    """Vega-Lite bars: similarity of each dose pill to each enrolled medicine, with the threshold as a rule."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": rows},
        "facet": {"row": {"field": "dose", "type": "nominal", "title": None, "sort": ["their metformin"],
                          "header": {"labelAngle": 0, "labelAlign": "left", "labelFontSize": 13,
                                     "labelFontWeight": "bold"}}},
        "spec": {"width": 380, "height": 110, "layer": [
            {"mark": {"type": "bar", "cornerRadiusEnd": 3},
             "encoding": {"y": {"field": "medicine", "type": "nominal", "title": None, "sort": None},
                          "x": {"field": "similarity", "type": "quantitative", "scale": {"domain": [0, 1]},
                                "title": "cosine similarity to the medicine's prototype"},
                          "color": {"field": "passes", "type": "nominal", "title": "above threshold",
                                    "scale": {"domain": [True, False], "range": ["#2a78d6", "#c9c8c3"]}},
                          "tooltip": [{"field": "medicine"}, {"field": "similarity"}]}},
            {"mark": {"type": "text", "align": "left", "dx": 4, "fontSize": 12},
             "encoding": {"y": {"field": "medicine", "type": "nominal", "sort": None},
                          "x": {"field": "similarity", "type": "quantitative"},
                          "text": {"field": "similarity", "type": "quantitative", "format": ".3f"}}},
            {"mark": {"type": "rule", "color": "#eb6834", "strokeWidth": 2},
             "encoding": {"x": {"datum": threshold}}},
        ]},
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12}, "legend": {"orient": "bottom",
                                                                               "labelFontSize": 12}},
    }


def outcome_chart(rows):
    """Vega-Lite stacked bars of what the patient sees (test results, run 4)."""
    ends = {}
    for r in sorted(rows, key=lambda r: r["k"]):  # label positions: the middle of each segment
        start = ends.get(r["set"], 0.0)
        r["mid"], ends[r["set"]] = start + r["percent"] / 2, start + r["percent"]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": 420, "height": 110,
        "data": {"values": rows},
        "layer": [
            {"mark": {"type": "bar", "stroke": "white", "strokeWidth": 1.5},
             "encoding": {"y": {"field": "set", "type": "nominal", "title": None, "sort": None},
                          "x": {"field": "percent", "type": "quantitative", "stack": "zero", "title": "% of own pills",
                                "scale": {"domain": [0, 100]}},
                          "color": {"field": "outcome", "type": "nominal", "sort": None, "title": None,
                                    "scale": {"range": ["#2a78d6", "#1baf7a", "#eda100", "#eb6834"]}},
                          "order": {"field": "k"},
                          "tooltip": [{"field": "set"}, {"field": "outcome"}, {"field": "percent"}]}},
            {"transform": [{"filter": "datum.percent >= 9"}],
             "mark": {"type": "text", "color": "white", "fontSize": 12, "fontWeight": "bold"},
             "encoding": {"y": {"field": "set", "type": "nominal", "sort": None},
                          "x": {"field": "mid", "type": "quantitative"},
                          "text": {"field": "percent", "type": "quantitative", "format": ".1f"}}},
        ],
        "config": {"axis": {"labelFontSize": 12, "titleFontSize": 12, "labelLimit": 260},
                   "legend": {"orient": "bottom", "labelFontSize": 12, "columns": 4}},
    }


def top_rows(look, score, catalogue, n):
    """The n best products as table rows, by look alone (score None) or after re-ranking."""
    ranked = np.argsort(-(look if score is None else score))[:n]
    return [{"rank": r + 1, "product": catalogue[j]["name"][:46], "imprint": catalogue[j]["imprint"],
             "look": round(float(look[j]), 4), **({} if score is None else {"with imprint": round(float(score[j]), 4)})}
            for r, j in enumerate(ranked)]
