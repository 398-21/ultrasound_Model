"""
Text reading with Florence-2 (MIT licence; microsoft/Florence-2, via transformers): pill imprints (the pill region of
a crop in four rotations, since pills lie at any angle; match.py turns the strings into evidence), the text on a box,
label or prescription (lines, top to bottom; core/label.py finds the medicines in them), and a one-line caption of a
photo (the input check, core/inference/input_type.py).

Each kind of reading is prepared (images -> model inputs), generated, then finished (tokens -> text), so a request
can run all its readings, and the pill model, in one GPU call (serving/app/vision.py).
"""
import torch

ROTATIONS = (0, 90, 180, 270)
# Reading settings (scripts/evaluate_imprint_reading.py, 400 validation crops framed as the API frames them, 13 Oct):
# own imprint token read on 41.5% with base at the central 70%, 45.2% at 92%, 50.0% with large at 92% (engraved:
# 37% -> 48%); false reads 2.5-3.0% throughout. Large takes ~180 ms per pill on a GPU, so it runs on the GPU only;
# on the CPU the base model reads (when it fits the time budget). Both share one preprocessor.
READ_CENTRE = 0.92
GPU_MODEL, CPU_MODEL = "florence-community/Florence-2-large", "florence-community/Florence-2-base"
PRESETS = {  # for the evaluation caches (scripts/read_imprints_cache.py)
    "base_c70": {"model": CPU_MODEL, "centre": 0.7, "served_framing": False},
    "large_c92": {"model": GPU_MODEL, "centre": READ_CENTRE, "served_framing": True},
    # the CPU reader as served (most live calls: the free GPU quota is per caller); read here on the GPU in fp16
    "base_c92": {"model": CPU_MODEL, "centre": READ_CENTRE, "served_framing": True},
    # large with the imprint adapter trained by scripts/train_imprint_reader.py (imprint reads only)
    "large_lora_v1": {"model": GPU_MODEL, "centre": READ_CENTRE, "served_framing": True,
                      "adapter": "outputs/imprint_reader/large_lora_v1/adapter.pt"},
    # v2 also learnt to write nothing (unreadable photos, wrong rotations); the better epoch on validation identify
    "large_lora_v2": {"model": GPU_MODEL, "centre": READ_CENTRE, "served_framing": True,
                      "adapter": "outputs/imprint_reader/large_lora_v2/adapter_best.pt"},
}
TOKENS = {"read": 20, "caption": 40, "text": 512}


class ImprintReader:
    def __init__(self, model=CPU_MODEL, device=None, beams=None, quantize=False, adapter=None):
        """quantize: int8 linear layers on the CPU (about 2x faster there); adapter: an imprint adapter for this
        model (core/imprint/lora.py), used for imprint reads only."""
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Florence2ForConditionalGeneration.from_pretrained(model, torch_dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.adapter = None
        if adapter:
            from epills.core.imprint.lora import load_lora
            self.adapter = load_lora(self.model, adapter)
        if quantize and self.device.type == "cpu":
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.beams = beams or (3 if self.device.type == "cuda" else 1)  # beams barely help, and cost most on CPU
        self.name = model

    @staticmethod
    def pill_region(crop, centre):
        """The central `centre` share of a crop (contract crops: the pill fills ~65%; tight crops: use 1.0)."""
        if centre >= 1:
            return crop
        w, h = crop.size
        dx, dy = w * (1 - centre) / 2, h * (1 - centre) / 2
        return crop.crop((int(dx), int(dy), int(w - dx), int(h - dy)))

    # ---- prepare: images -> a job (model inputs + what's needed to finish) ---------------------------------------
    def _job(self, task, prompt, images, **extra):
        x = self.processor(text=[prompt] * len(images), images=images, return_tensors="pt")
        return {"task": task, "ids": x["input_ids"], "px": x["pixel_values"], "tokens": TOKENS[task],
                "images": len(images), **extra}

    def prepare_read(self, crops, centre=READ_CENTRE):
        images = [self.pill_region(c.convert("RGB"), centre).rotate(a, expand=True) for c in crops for a in ROTATIONS]
        return self._job("read", "<OCR>", images, crops=len(crops))

    def prepare_caption(self, image):
        return self._job("caption", "<CAPTION>", [image.convert("RGB")])

    def prepare_text(self, image):
        """A tall page is read in overlapping bands: Florence sees every image at 768 x 768."""
        image = image.convert("RGB")
        w, h = image.size
        if h <= 1.25 * w:
            bands = [image]
        else:
            band = int(w * 0.8)
            bands = [image.crop((0, top, w, min(h, top + band)))
                     for top in range(0, max(1, h - band // 6), int(band * 0.85))]
        return self._job("text", "<OCR_WITH_REGION>", bands, sizes=[b.size for b in bands])

    # ---- generate and finish ---------------------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, input_ids, pixel_values, max_new_tokens, task="read"):
        """Model inputs (CPU tensors) -> generated token ids (CPU), with this reader's own model."""
        if self.adapter:
            from epills.core.imprint.lora import set_lora
            set_lora(self.model, task == "read")
        out = self.model.generate(input_ids=input_ids.to(self.device),
                                  pixel_values=pixel_values.to(self.device, self.dtype),
                                  max_new_tokens=max_new_tokens, num_beams=self.beams)
        return out.cpu()

    def finish(self, job, ids):
        if job["task"] == "read":
            texts = self.processor.batch_decode(ids, skip_special_tokens=True)
            n = len(ROTATIONS)
            return [texts[i:i + n] for i in range(0, len(texts), n)]
        if job["task"] == "caption":
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        lines = []
        for raw, size in zip(self.processor.batch_decode(ids, skip_special_tokens=False), job["sizes"]):
            found = self.processor.post_process_generation(raw, task="<OCR_WITH_REGION>", image_size=size)
            found = found["<OCR_WITH_REGION>"]
            band = sorted((min(q[1::2]), min(q[0::2]), t.replace("</s>", "").strip())
                          for q, t in zip(found["quad_boxes"], found["labels"]))
            for _, _, text in band:
                if text and text not in lines[-8:]:  # band overlaps read some lines twice
                    lines.append(text)
        return lines

    def run(self, job, batch=32):
        """Generate in batches of images (memory), then finish."""
        rows = [row for i in range(0, len(job["ids"]), batch)
                for row in self.generate(job["ids"][i:i + batch], job["px"][i:i + batch], job["tokens"], job["task"])]
        return self.finish(job, rows)

    # ---- one-call conveniences (scripts, tests) --------------------------------------------------------------------
    def read(self, crops, centre=READ_CENTRE):
        """PIL crops -> per crop, its reads in the four rotations."""
        return self.run(self.prepare_read(crops, centre)) if crops else []

    def caption(self, image):
        return self.run(self.prepare_caption(image))

    def read_text(self, image):
        return self.run(self.prepare_text(image))
