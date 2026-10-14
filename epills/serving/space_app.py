"""
The Hugging Face Space entry point (Gradio SDK on ZeroGPU hardware, free for a personal account): the REST API of
serving/app/main.py unchanged, plus a demo page at /ui. scripts/deploy_hf.py uploads this file as the Space's app.py.

ZeroGPU lends a GPU per call. A request needs at most one (serving/app/vision.py): the pill model and every reading
(Florence-2-large: imprints, labels, the input check) run together in one @spaces.GPU call; requests that only embed
pills stay on the CPU, so the free allowance (a daily number of runs) lasts. When no GPU is granted (allowance used
up, queue too long) the request runs on the CPU with Florence-2-base within a time budget, and stays on the CPU for
COOLDOWN seconds, so answers keep coming.
Off ZeroGPU it is the plain CPU API with the demo page:  python serving/space_app.py   (http://localhost:7860/ui)
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE if (HERE / "serving").is_dir() else HERE.parent  # the Space root (app.py), or the epills folder
sys.path[:0] = [str(BASE / "src"), str(BASE)]

try:
    import spaces  # before torch: on ZeroGPU it patches torch
except ImportError:
    spaces = None

import gradio as gr
import torch
import uvicorn
from fastapi import HTTPException
from fastapi.responses import RedirectResponse

import threading

from epills.core.imprint.ocr import GPU_MODEL
from epills.core.inference.pill_finder import find_pills
from epills.core.inference.runtime import cpu_cores
from serving.app.fallback import GpuOrCpu
from serving.app.main import IDENTIFY_NOTE, LABEL_NOTE, app, asset, identify_pill, imprint_adapter_path, vision

ZERO_GPU = spaces is not None and os.environ.get("SPACES_ZERO_GPU", "").lower() in ("1", "t", "true")
COOLDOWN = 600  # seconds on the CPU after ZeroGPU refused a call

if os.environ.get("EPILLS_MODEL_REPO"):
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import whoami
        who = whoami(token)["name"] if token else "no token"
    except Exception as error:
        who = f"token rejected ({type(error).__name__})"
    print(f"assets from {os.environ['EPILLS_MODEL_REPO']}: HF_TOKEN {'set' if token else 'NOT set'}, {who}", flush=True)
for name in ("safety_db", "embedder", "catalogue", "reader"):  # now: ZeroGPU needs the models before start-up
    try:
        asset(name)
    except HTTPException as error:
        print(f"{name}: {error.detail}", flush=True)
embedder, reader = app.state.embedder, app.state.reader
print(f"CPU: {cpu_cores()} usable cores (os.cpu_count {os.cpu_count()}); torch threads {torch.get_num_threads()}",
      flush=True)
pipeline = vision() if embedder is not None else None
if pipeline is not None and reader is not None:
    pipeline.warm_up()  # times a CPU reading, for the CPU time budget
    print(f"CPU reading: {pipeline.seconds_per_image:.2f} s per image; budget {pipeline.budget:.0f} s", flush=True)
    threading.Thread(target=lambda: asset("label_reader").vocab, daemon=True).start()  # RxNorm names, ~3 s

if ZERO_GPU and pipeline is not None:
    from transformers import Florence2ForConditionalGeneration
    gpu_model = copy.deepcopy(embedder.model).to("cuda")  # placed now; ZeroGPU moves them to a real GPU per call
    gpu_ocr = Florence2ForConditionalGeneration.from_pretrained(GPU_MODEL, torch_dtype=torch.float16).eval()
    adapter = imprint_adapter_path()  # trained on pill imprints; switched on for imprint reads only
    if adapter is not None:  # added on the CPU, then moved with the model
        from epills.core.imprint.lora import load_lora, set_lora
        app.state.gpu_imprint_adapter = load_lora(gpu_ocr, adapter).get("name", "imprint adapter")
        print(f"imprint adapter: {app.state.gpu_imprint_adapter}", flush=True)
    gpu_ocr = gpu_ocr.to("cuda")

    @spaces.GPU(duration=30)
    def gpu_compute(payload):
        """One request's work in one GPU call: the pill model, then every reading (large model)."""
        out = {"embed": None, "gen": [], "device": "zerogpu"}
        with torch.no_grad():
            if payload["pixels"] is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out["embed"] = gpu_model(payload["pixels"].cuda())["embedding"].float().cpu().numpy()
            for job in payload["jobs"]:
                if adapter is not None:
                    set_lora(gpu_ocr, job["task"] == "read")
                out["gen"].append(gpu_ocr.generate(input_ids=job["ids"].cuda(), pixel_values=job["px"].cuda().half(),
                                                   max_new_tokens=job["tokens"], num_beams=3).cpu())
        return out

    pipeline.compute = GpuOrCpu(gpu_compute, pipeline.cpu_compute, COOLDOWN)


# ---- demo page -------------------------------------------------------------------------------------------------------
def candidates(vectors, imprint=None, top_k=3, reads=None):
    try:
        return identify_pill(vectors, imprint=imprint or None, top_k=top_k, reads=reads)["candidates"]
    except HTTPException as error:
        raise gr.Error(str(error.detail))


def label(c):
    read = ", imprint read" if c.get("imprint_read") else ""
    return f"{c['name']} (imprint {c['imprint'] or 'none'}{read}), similarity {c['similarity']:.2f}"


def ui_identify(photo, imprint):
    if photo is None or embedder is None:
        raise gr.Error("Add a photo of one pill." if embedder else "The model isn't loaded; see /health.")
    cat, out = app.state.catalogue, []
    crops = embedder.frame(photo)
    r = vision().run(crops, None if imprint else crops)
    for c in candidates(r["embeddings"], imprint, top_k=6, reads=(r["reads"] or [None])[0]):
        images = cat.product(c["pill_id"])["images"]
        if images:
            out.append((str(cat.image_file(c["pill_id"], images[0]["n"])), label(c)))
    return out


def ui_scan(photo):
    if photo is None or embedder is None:
        raise gr.Error("Add a photo of the pills." if embedder else "The model isn't loaded; see /health.")
    boxes = find_pills(photo)
    if not boxes:
        raise gr.Error("No pill found. Spread the pills apart on a plain background and photograph them again.")
    rows, marks = [], []
    crops = embedder.frame(photo, boxes)
    r = vision().run(crops, crops)
    for k, (box, vector, read) in enumerate(zip(boxes, r["embeddings"], r["reads"] or [None] * len(crops)), 1):
        rows.append([k] + [label(c) for c in candidates(vector[None], reads=read)])
        marks.append(([round(v) for v in box], f"pill {k}"))
    return (photo, marks), rows


def ui_label(photo):
    if photo is None or reader is None:
        raise gr.Error("Add a photo of a box or label." if reader else "Text reading isn't loaded; see /health.")
    try:
        found = asset("label_reader").read(vision().run(text=photo)["lines"] or [])
    except Exception as error:
        raise gr.Error(f"Couldn't look the names up: {error}")
    if not found:
        raise gr.Error("No medicine name found. Photograph the label closer, straight on and in good light.")
    return [[c["name"], ", ".join(i["name"] for i in c["ingredients"]), c["rxcui"], c["rxnorm_name"] or ""]
            for c in found]


with gr.Blocks(title="PharmaScan") as demo:
    gr.Markdown("# PharmaScan\nA demo of the PharmaScan API ([REST API docs](/docs)). " + IDENTIFY_NOTE)
    with gr.Tab("Identify a pill"):
        with gr.Row():
            with gr.Column():
                pill = gr.Image(type="pil", label="One pill in the middle of the photo, filling about 70% of it")
                imprint = gr.Textbox(label="Imprint (optional)", placeholder="e.g. M 547")
                identify_button = gr.Button("Identify", variant="primary")
            gallery = gr.Gallery(label="Most alike pills", columns=3)
        identify_button.click(ui_identify, [pill, imprint], gallery)
    with gr.Tab("Scan a photo"):
        with gr.Row():
            with gr.Column():
                full = gr.Image(type="pil", label="Several pills, apart, on a plain background")
                scan_button = gr.Button("Scan", variant="primary")
            found = gr.AnnotatedImage(label="Pills found")
        table = gr.Dataframe(headers=["pill", "most alike", "2nd", "3rd"], label="Catalogue matches")
        scan_button.click(ui_scan, full, [found, table])
    with gr.Tab("Read a box or label"):
        with gr.Row():
            with gr.Column():
                box_photo = gr.Image(type="pil", label="The medicine's box or pharmacy label, close up")
                label_button = gr.Button("Read", variant="primary")
            medicines = gr.Dataframe(headers=["medicine", "ingredients", "RxNorm code", "RxNorm name"],
                                     label=LABEL_NOTE)
        label_button.click(ui_label, box_photo, medicines)

app = gr.mount_gradio_app(app, demo, path="/ui", ssr_mode=False)  # SSR's Node server would take port 7860


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


if __name__ == "__main__":
    if ZERO_GPU and pipeline is not None:
        from spaces.zero import startup  # what demo.launch() would run: hands the placed weights to ZeroGPU
        startup()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
