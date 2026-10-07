"""
Stage 2 training data: the split manifest (scripts/make_splits.py), augmentations and the P x K batch sampler.

Images are the 384 px C3PI crops; the model sees 224 px. Labels:
  label   appearance group (the metric-learning class)
  shape   one of the C3PI shapes (auxiliary softmax head)
  colour  multi-hot over the C3PI colours; two-tone capsules have two (auxiliary multi-label head)
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import v2 as T

IMAGE_SIZE = 224


def train_transform(colour_jitter=0.05, size=IMAGE_SIZE):
    """
    The plan's augmentations: full 0-360 degree rotation, flips, blur and perspective, with only light colour jitter
    because colour identifies pills. A mild crop/scale change covers framing differences between cropper and
    Stage 1. Output: float tensor in [0, 1].
    """
    ops = [
        T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
        T.RandomRotation(180, interpolation=T.InterpolationMode.BILINEAR, fill=None),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
    ]
    if colour_jitter > 0:
        ops.append(T.ColorJitter(brightness=2 * colour_jitter, contrast=2 * colour_jitter,
                                 saturation=colour_jitter, hue=colour_jitter / 4))
    return T.Compose([T.ToImage(), *ops, T.ToDtype(torch.float32, scale=True)])


def eval_transform(size=IMAGE_SIZE):
    return T.Compose([T.ToImage(), T.Resize(size, antialias=True), T.CenterCrop(size), T.ToDtype(torch.float32, scale=True)])


class Vocab:
    """Shape and colour vocabularies from the training split (sorted, so they are stable across runs)."""

    def __init__(self, manifest):
        self.shapes = sorted(set(manifest["shape"]) - {""})
        self.colours = sorted({c for v in manifest["color"] for c in str(v).split(";")} - {""})

    def shape_index(self, shape):
        return self.shapes.index(shape) if shape in self.shapes else -1  # -1: no label, loss masked

    def colour_vector(self, colour):
        v = np.zeros(len(self.colours), np.float32)
        for c in str(colour).split(";"):
            if c in self.colours:
                v[self.colours.index(c)] = 1
        return v

    def to_dict(self):
        return {"shapes": self.shapes, "colours": self.colours}


class PillCrops(Dataset):
    """One split of the manifest. Returns dict(image, label (int), shape (int, -1 unknown), colour (multi-hot))."""

    def __init__(self, manifest, image_root, transform, vocab, label_index=None):
        self.df = manifest.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform
        self.label_index = label_index or {g: i for i, g in enumerate(sorted(self.df["label"].unique()))}
        self.labels = self.df["label"].map(self.label_index).to_numpy()
        self.shapes = np.array([vocab.shape_index(s) for s in self.df["shape"]])
        self.colours = np.stack([vocab.colour_vector(c) for c in self.df["color"]])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        with Image.open(self.image_root / self.df.at[i, "crop_path"]) as img:
            image = self.transform(img.convert("RGB"))
        return {"image": image, "label": int(self.labels[i]), "shape": int(self.shapes[i]),
                "colour": torch.from_numpy(self.colours[i]), "index": i}


class PKSampler(Sampler):
    """
    Batches of P labels x K images, so every batch holds valid positive pairs for the metric loss. Labels with fewer
    than K images are sampled with replacement. One epoch visits each image about once.
    """

    def __init__(self, labels, p=32, k=4, seed=0):
        self.by_label = {}
        for idx, lab in enumerate(labels):
            self.by_label.setdefault(int(lab), []).append(idx)
        self.p, self.k = min(p, len(self.by_label)), k
        self.n_batches = max(1, len(labels) // (self.p * k))
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        labels = list(self.by_label)
        for _ in range(self.n_batches):
            batch = []
            for lab in self.rng.sample(labels, self.p):
                pool = self.by_label[lab]
                batch += self.rng.sample(pool, self.k) if len(pool) >= self.k else self.rng.choices(pool, k=self.k)
            yield batch


def load_manifest(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
