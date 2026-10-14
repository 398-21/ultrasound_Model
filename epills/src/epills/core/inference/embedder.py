"""
The served Stage 2 model: pill crops -> unit-length embeddings, plus the calibration it was released with.

Loads a serving export (scripts/export_model.py) or a training best.pt. Crops follow the crop contract
(preprocess.crop_window); a crop that is already framed is passed as is.
"""

import numpy as np
import torch

from epills.core.inference.preprocess import CROP_SIZE, IMAGE_SIZE, WINDOW, crop_window, eval_transform
from epills.core.inference.runtime import cpu_cores
from epills.core.models.hybrid import HybridPillNet


class Embedder:
    def __init__(self, path, device=None, batch_size=16):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(cpu_cores())  # the container's real cores: oversubscribing slows PyTorch
        self.model, ckpt = HybridPillNet.from_checkpoint(path, self.device)
        self.version = ckpt.get("version") or ckpt.get("config", {}).get("name", "unversioned")
        self.calibration = ckpt.get("calibration") or {}
        self.window = self.calibration.get("window") or WINDOW  # the crop contract this model was calibrated with
        # embeddings are comparable only within one space: same model, same crop contract, loose-pill photos
        self.embedding_space_id = f"pill-crop:{self.version}:w{self.window:g}"
        self.preprocess_version = f"crop-contract-1:window-{self.window:g}:crop-{CROP_SIZE}:input-{IMAGE_SIZE}"
        self.transform = eval_transform()
        self.batch_size = batch_size

    def __call__(self, crops):
        """PIL crops (already framed by the crop contract) -> (n, d) float32 unit vectors."""
        out = [self.forward(torch.stack([self.transform(c.convert("RGB")) for c in crops[i:i + self.batch_size]]))
               for i in range(0, len(crops), self.batch_size)]
        return np.concatenate(out) if out else np.zeros((0, 256), np.float32)

    @torch.no_grad()
    def forward(self, x):
        """A batch of transformed crops (CPU tensor) -> (n, d) float32 array. The Space app swaps in a ZeroGPU one."""
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            emb = self.model(x.to(self.device))["embedding"]
        return emb.float().cpu().numpy()

    def frame(self, image, boxes=None):
        """A photo and optional boxes (x0, y0, x1, y1) -> crops by the crop contract; without boxes the photo is one."""
        return [crop_window(image, b, self.window) for b in boxes] if boxes else [image]

    def embed_photo(self, image, boxes=None):
        """A photo and optional boxes -> embeddings."""
        return self(self.frame(image, boxes))
