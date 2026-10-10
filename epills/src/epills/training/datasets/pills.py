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

from epills.core.inference.preprocess import IMAGE_SIZE, eval_transform  # noqa: F401 (the API's transform)


class WhiteBalance(torch.nn.Module):
    """Random red and blue gains (camera white balance) on a float image in [0, 1]."""

    def __init__(self, strength):
        super().__init__()
        self.strength = strength

    def forward(self, x):
        gain = 1 + (torch.rand(3) * 2 - 1) * self.strength
        gain[1] = 1.0
        return (x * gain[:, None, None]).clamp(0, 1)


class Downscale(torch.nn.Module):
    """Shrink to a random size in px, then back: a small, low-resolution pill (MEDISEG's median is 81 px)."""

    def __init__(self, px):
        super().__init__()
        self.px = px

    def forward(self, x):
        size = list(x.shape[-2:])
        s = int(torch.randint(self.px[0], self.px[1] + 1, ()).item())
        x = T.functional.resize(x, [s, s], antialias=True)
        return T.functional.resize(x, size, antialias=True)


class JitterWindow(torch.nn.Module):
    """
    A detector's box error on a contract crop (PIL): the window resized by a factor in `scale` and its centre shifted
    by up to `shift` x its side, edge pixels repeated outside. scale < 1 is a tight box, > 1 a loose one.
    """

    def __init__(self, scale, shift):
        super().__init__()
        self.scale, self.shift = scale, shift

    def forward(self, image):
        size = image.size[0]
        side = int(size * float(torch.empty(()).uniform_(*self.scale)))
        dx, dy = (torch.empty(2).uniform_(-self.shift, self.shift) * size).tolist()
        left, top = int(round((size - side) / 2 + dx)), int(round((size - side) / 2 + dy))
        arr = np.asarray(image)
        pad = max(0, -left, -top, left + side - size, top + side - size)
        if pad:
            arr = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            left, top = left + pad, top + pad
        return Image.fromarray(arr[top:top + side, left:left + side])


class SensorNoise(torch.nn.Module):
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, x):
        return (x + torch.randn_like(x) * torch.rand(()) * self.std).clamp(0, 1)


def train_transform(colour_jitter=0.05, size=IMAGE_SIZE, crop_jitter=None, photo=None, **_):
    """
    The plan's augmentations: full 0-360 degree rotation, flips, blur and perspective, with only light colour jitter
    because colour identifies pills. Output: float tensor in [0, 1].
      crop_jitter  None: a mild crop/scale change (runs 1-3). {scale, shift}: the window resized and shifted the
                   way a detector's box errs (run 4), see JitterWindow.
      photo        None: off (runs 1-3). {brightness, contrast, white_balance, downscale_p, downscale_px, jpeg_p,
                   jpeg_quality, noise_p, noise_std}: other phones and light (run 4). Hue stays nearly fixed.
    """
    pre, framing = [], [T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True)]
    if crop_jitter:  # on the PIL crop, before it becomes a tensor
        pre, framing = [JitterWindow(crop_jitter["scale"], crop_jitter["shift"])], [T.Resize(size, antialias=True)]
    ops = framing + [
        T.RandomRotation(180, interpolation=T.InterpolationMode.BILINEAR, fill=None),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.3),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3),
    ]
    if colour_jitter > 0 or photo:
        ops.append(T.ColorJitter(brightness=photo["brightness"] if photo else 2 * colour_jitter,
                                 contrast=photo["contrast"] if photo else 2 * colour_jitter,
                                 saturation=colour_jitter, hue=colour_jitter / 4))
    after = []
    if photo:
        ops += [T.RandomApply([Downscale(photo["downscale_px"])], p=photo["downscale_p"]),
                T.RandomApply([T.JPEG(tuple(photo["jpeg_quality"]))], p=photo["jpeg_p"])]
        after = [WhiteBalance(photo["white_balance"]),
                 T.RandomApply([SensorNoise(photo["noise_std"])], p=photo["noise_p"])]
    return T.Compose([*pre, T.ToImage(), *ops, T.ToDtype(torch.float32, scale=True), *after])


def paste_neighbour(image, other, rng):
    """
    Another pill partly in the frame (pills side by side, or a loose detector box): the centre of another crop,
    with a soft round mask, pasted half outside a random edge of this crop. Both are PIL crops of one size.
    """
    size = image.size[0]
    side = int(size * rng.uniform(0.35, 0.55))
    c, half = other.size[0] // 2, int(other.size[0] * 0.3)
    patch = other.crop((c - half, c - half, c + half, c + half)).resize((side, side))
    yy, xx = np.mgrid[:side, :side]
    r = np.hypot(xx - side / 2, yy - side / 2) / (side / 2)
    mask = Image.fromarray((np.clip((1.0 - r) / 0.15, 0, 1) * 255).astype(np.uint8))
    along = int(rng.uniform(0.15, 0.85) * size) - side // 2
    out = -(side // 2)
    pos = [(out, along), (size + out, along), (along, out), (along, size + out)][int(rng.integers(4))]
    image = image.copy()
    image.paste(patch, pos, mask)
    return image


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
    """
    One split of the manifest. Returns dict(image, label (int), shape (int, -1 unknown), colour (multi-hot)).
    neighbour_p: share of crops that get another medicine's pill pasted partly into the frame (run 4).
    """

    def __init__(self, manifest, image_root, transform, vocab, label_index=None, neighbour_p=0.0):
        self.df = manifest.reset_index(drop=True)
        self.neighbour_p = neighbour_p
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
            image = img.convert("RGB")
        if self.neighbour_p and random.random() < self.neighbour_p:
            j = random.randrange(len(self.df))
            if self.labels[j] != self.labels[i]:
                with Image.open(self.image_root / self.df.at[j, "crop_path"]) as other:
                    other = other.convert("RGB").resize(image.size)
                image = paste_neighbour(image, other, np.random.default_rng(random.getrandbits(32)))
        image = self.transform(image)
        return {"image": image, "label": int(self.labels[i]), "shape": int(self.shapes[i]),
                "colour": torch.from_numpy(self.colours[i]), "index": i}


class PKSampler(Sampler):
    """
    Batches of P labels x K images, so every batch holds valid positive pairs for the metric loss. Labels with fewer
    than K images are sampled with replacement. One epoch visits each image about once.
    weighting: "uniform" picks every label equally often; "sqrt" in proportion to the square root of its photo count,
    so thousands of 2-photo reference-only classes don't crowd out the everyday photos.
    is_reference: per image, whether it is a studio reference; when given, a label that has both kinds contributes
    K/2 references and K/2 everyday photos, so every such label pairs the two in the batch.
    """

    def __init__(self, labels, p=32, k=4, seed=0, weighting="uniform", is_reference=None):
        self.by_label = {}
        for idx, lab in enumerate(labels):
            self.by_label.setdefault(int(lab), []).append(idx)
        self.p, self.k = min(p, len(self.by_label)), k
        self.n_batches = max(1, len(labels) // (self.p * k))
        self.rng = random.Random(seed)
        self.labels = list(self.by_label)
        self.weights = None
        if weighting == "sqrt":
            w = np.sqrt([len(self.by_label[lab]) for lab in self.labels])
            self.weights = w / w.sum()
        self.np_rng = np.random.default_rng(seed)
        self.split = {}
        if is_reference is not None:
            for lab, pool in self.by_label.items():
                refs = [i for i in pool if is_reference[i]]
                everyday = [i for i in pool if not is_reference[i]]
                if refs and everyday:
                    self.split[lab] = (refs, everyday)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            chosen = (self.rng.sample(self.labels, self.p) if self.weights is None else
                      [self.labels[i] for i in self.np_rng.choice(len(self.labels), self.p, replace=False,
                                                                   p=self.weights)])
            for lab in chosen:
                if lab in self.split:
                    refs, everyday = self.split[lab]
                    batch += self.draw(refs, self.k // 2) + self.draw(everyday, self.k - self.k // 2)
                else:
                    batch += self.draw(self.by_label[lab], self.k)
            yield batch


    def draw(self, pool, n):
        return self.rng.sample(pool, n) if len(pool) >= n else self.rng.choices(pool, k=n)


def load_manifest(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)
