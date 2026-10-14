"""
Where a request's models run (latency target: every call under 5 s).

At most one GPU call per request: the pill model and every reading the request needs (imprints, a caption for the
input check, label text) are prepared on the CPU, run together in `compute`, then finished. Requests that only need
embeddings never use the GPU (0.2-0.5 s per pill on the CPU), so the free ZeroGPU allowance goes to reading.

Without a GPU (none here, or ZeroGPU refused), readings run on the CPU with the small model while they fit the time
budget (EPILLS_CPU_READ_BUDGET seconds, default 3); the rest are skipped and reported as "skipped", never dropped
silently. Label text is always read (the request is pointless without it), even when slow. The CPU cost per image
is measured at start-up (warm_up).
"""
import os
import time

import numpy as np
import torch

COST = {"read": 1.0, "caption": 1.5, "text": 6.0}  # CPU cost per image, relative to one imprint rotation


class Vision:
    def __init__(self, embedder, reader=None):
        self.embedder, self.reader = embedder, reader
        self.budget = float(os.environ.get("EPILLS_CPU_READ_BUDGET", "3"))
        self.seconds_per_image = None

    def warm_up(self):
        """Time one CPU imprint reading, so the budget can tell what fits."""
        if self.reader is None:
            return
        from PIL import Image
        self.seconds_per_image = None
        self._timed_generate(self.reader.prepare_read([Image.new("RGB", (384, 384), (200, 200, 200))]))

    def _cpu_cost(self, job):
        """Expected CPU seconds for a job; 0 until a reading has been timed (then measured, never guessed)."""
        return (self.seconds_per_image or 0.0) * job["images"] * COST[job["task"]]

    def _timed_generate(self, job):
        start = time.time()
        ids = self.reader.generate(job["ids"], job["px"], job["tokens"], job["task"])
        per_image = (time.time() - start) / max(1, job["images"] * COST[job["task"]])
        self.seconds_per_image = per_image if self.seconds_per_image is None else \
            0.7 * self.seconds_per_image + 0.3 * per_image  # follows the machine's real speed
        return ids

    def compute(self, payload):
        """The Space replaces this with a ZeroGPU call that falls back to cpu_compute (serving/space_app.py)."""
        return self.cpu_compute(payload)

    def cpu_compute(self, payload):
        out = {"embed": None, "gen": [], "device": "cpu"}
        if payload["pixels"] is not None:
            out["embed"] = self.embedder.forward(payload["pixels"])
        spent = 0.0
        for job in payload["jobs"]:
            cost = self._cpu_cost(job)
            if job["task"] == "text" or spent + cost <= self.budget:
                out["gen"].append(self._timed_generate(job))
                spent += cost
            else:
                out["gen"].append(None)
        return out

    def run(self, embed=None, read=None, caption=None, text=None):
        """
        embed, read: lists of crops; caption: one photo; text: one image or a list (pages). Returns embeddings,
        reads (per crop), caption, lines, `status` per reading (used / skipped / unavailable / not_requested) and the
        device the request ran on.
        """
        status = {"read": "not_requested", "caption": "not_requested", "text": "not_requested"}
        wanted = {"read": bool(read), "caption": caption is not None, "text": bool(text)}
        jobs = []
        if self.reader is None:
            status.update({k: "unavailable" for k, w in wanted.items() if w})
        else:  # the input check first: it is a safety check, and gets the CPU budget before imprints
            if wanted["caption"]:
                jobs.append(self.reader.prepare_caption(caption))
            if wanted["read"]:
                jobs.append(self.reader.prepare_read(read))
            if wanted["text"]:
                jobs += [self.reader.prepare_text(page) for page in (text if isinstance(text, list) else [text])]
        pixels = torch.stack([self.embedder.transform(c.convert("RGB")) for c in embed]) if embed else None
        if jobs:
            out = self.compute({"pixels": pixels, "jobs": jobs})
        else:
            out = {"embed": self.embedder.forward(pixels) if pixels is not None else None, "gen": [], "device": "cpu"}
        res = {"embeddings": None if out["embed"] is None else np.asarray(out["embed"], np.float32),
               "reads": None, "caption": None, "lines": None, "status": status, "device": out.get("device", "cpu")}
        lines = []
        for job, ids in zip(jobs, out["gen"]):
            if ids is None:
                status[job["task"]] = "skipped"
                continue
            status[job["task"]] = "used"
            done = self.reader.finish(job, ids)
            if job["task"] == "read":
                res["reads"] = done
            elif job["task"] == "caption":
                res["caption"] = done
            else:
                lines += done
        if status["text"] == "used":
            res["lines"] = lines
        return res
