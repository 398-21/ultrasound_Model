"""
Imprint reading with Florence-2 (MIT licence; microsoft/Florence-2, via transformers): the pill region of a crop, read
in four rotations (pills lie at any angle). Returns the raw strings; match.py turns them into evidence.

Measured on 150 validation crops (2026-10-10, base model): a token of the pill's own imprint is read on 43% of
pills (printed 63%, debossed 36%; C3PI "Clear" 74%, "Can not Identify" 0%), a random other pill's on 0.7%;
0.28 s per pill on the GPU (4 rotations). The large model reads 47% at twice the time.
"""
import torch

ROTATIONS = (0, 90, 180, 270)


class ImprintReader:
    def __init__(self, model="florence-community/Florence-2-base", device=None, batch_size=32):
        from transformers import AutoProcessor, Florence2ForConditionalGeneration
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Florence2ForConditionalGeneration.from_pretrained(model, torch_dtype=self.dtype)
        self.model = self.model.to(self.device).eval()
        self.name, self.batch_size = model, batch_size

    @staticmethod
    def pill_region(crop, centre):
        """The central `centre` share of a crop (contract crops: the pill fills ~60-70%; tight crops: use 1.0)."""
        if centre >= 1:
            return crop
        w, h = crop.size
        dx, dy = w * (1 - centre) / 2, h * (1 - centre) / 2
        return crop.crop((int(dx), int(dy), int(w - dx), int(h - dy)))

    @torch.no_grad()
    def read(self, crops, centre=0.7):
        """PIL crops -> per crop, its reads in the four rotations."""
        images = [self.pill_region(c.convert("RGB"), centre).rotate(a, expand=True) for c in crops for a in ROTATIONS]
        texts = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            x = self.processor(text=["<OCR>"] * len(batch), images=batch, return_tensors="pt")
            x = x.to(self.device, self.dtype)
            out = self.model.generate(input_ids=x["input_ids"], pixel_values=x["pixel_values"], max_new_tokens=20,
                                      num_beams=3)
            texts += self.processor.batch_decode(out, skip_special_tokens=True)
        n = len(ROTATIONS)
        return [texts[i:i + n] for i in range(0, len(texts), n)]
