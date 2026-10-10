"""
The Hugging Face Space entry point (Gradio SDK on ZeroGPU hardware, free for a personal account): the REST API of
serving/app/main.py unchanged, plus a demo page at /ui. scripts/deploy_hf.py uploads this file as the Space's app.py.

ZeroGPU lends a GPU per call. The model's forward pass asks for one (@spaces.GPU); when none is granted (daily quota
used up, queue too long) it runs on the CPU instead and stays there for COOLDOWN seconds, so answers keep coming.
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

from epills.core.inference.pill_finder import find_pills
from serving.app.fallback import GpuOrCpu
from serving.app.main import IDENTIFY_NOTE, app, asset, identify_pill

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
for name in ("safety_db", "embedder", "catalogue"):  # now, not on first use: ZeroGPU needs the model before start-up
    try:
        asset(name)
    except HTTPException as error:
        print(f"{name}: {error.detail}", flush=True)
embedder = app.state.embedder

if ZERO_GPU and embedder is not None:
    gpu_model = copy.deepcopy(embedder.model).to("cuda")  # placed now; ZeroGPU moves it to a real GPU per call

    @spaces.GPU(duration=20)
    def gpu_forward(x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            return gpu_model(x.cuda())["embedding"].float().cpu().numpy()

    embedder.forward = GpuOrCpu(gpu_forward, embedder.forward, COOLDOWN)


# ---- demo page -------------------------------------------------------------------------------------------------------
def candidates(vectors, imprint=None, top_k=3):
    try:
        return identify_pill(vectors, imprint=imprint or None, top_k=top_k)["candidates"]
    except HTTPException as error:
        raise gr.Error(str(error.detail))


def label(c):
    return f"{c['name']} (imprint {c['imprint'] or 'none'}), similarity {c['similarity']:.2f}"


def ui_identify(photo, imprint):
    if photo is None or embedder is None:
        raise gr.Error("Add a photo of one pill." if embedder else "The model isn't loaded; see /health.")
    cat, out = app.state.catalogue, []
    for c in candidates(embedder.embed_photo(photo), imprint, top_k=6):
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
    for k, (box, vector) in enumerate(zip(boxes, embedder.embed_photo(photo, boxes)), 1):
        rows.append([k] + [label(c) for c in candidates(vector[None])])
        marks.append(([round(v) for v in box], f"pill {k}"))
    return (photo, marks), rows


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

app = gr.mount_gradio_app(app, demo, path="/ui", ssr_mode=False)  # SSR's Node server would take port 7860


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


if __name__ == "__main__":
    if ZERO_GPU and embedder is not None:
        from spaces.zero import startup  # what demo.launch() would run: hands the placed weights to ZeroGPU
        startup()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
