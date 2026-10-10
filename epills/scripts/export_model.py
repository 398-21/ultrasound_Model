"""
Export a trained, calibrated Stage 2 model for the API (serving/): one file with half-precision weights (half the
size), the ViT config (no download at start-up), the vocabulary, the calibration (threshold and margin, which
belong to this model only) and a version string. The API reports the version in every answer; embeddings made by
another version must not be mixed with it (the app re-embeds its galleries, the catalogue is rebuilt).

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\export_model.py --run outputs\\runs\\stage2_run3
  -> outputs/serving/stage2_model.pt  (upload to the Hugging Face model repo as stage2_model.pt)
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.models.hybrid import HybridPillNet  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=EPILLS_ROOT / "outputs" / "serving" / "stage2_model.pt")
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else EPILLS_ROOT / args.run

    model, ckpt = HybridPillNet.from_checkpoint(run / "best.pt")
    calibration = json.loads((run / "calibration.json").read_text())
    trained = datetime.date.fromtimestamp((run / "best.pt").stat().st_mtime).isoformat()
    mcfg, vocab = ckpt["config"]["model"], ckpt["vocab"]
    export = {
        "format": "epills-stage2/1",
        "version": f"{run.name}@{trained}",
        "model_args": {"embedding_dim": mcfg["embedding_dim"], "refine_pool": mcfg["refine_pool"],
                       "num_shapes": len(vocab["shapes"]) if mcfg["aux_heads"] else 0,
                       "num_colours": len(vocab["colours"]) if mcfg["aux_heads"] else 0},
        "vit_config": model.vit.config.to_dict(),
        "state_dict": {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
        "vocab": vocab,
        "calibration": {k: calibration.get(k) for k in ("threshold", "margin", "window", "rule", "calibrated_on")},
        "validation": ckpt["metrics"],
        "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export, args.out)

    # the half-precision copy must give the same embeddings
    reloaded, _ = HybridPillNet.from_checkpoint(args.out)
    x = torch.rand(4, 3, 224, 224)
    with torch.no_grad():
        a, b = model(x)["embedding"], reloaded(x)["embedding"]
    cos = float((a * b).sum(dim=1).min())
    print(f"{args.out} ({args.out.stat().st_size / 1e6:.0f} MB), version {export['version']}, "
          f"threshold {calibration['threshold']:.4f}, margin {calibration['margin']:.3f}, "
          f"window {calibration.get('window') or 'default'}; "
          f"half vs full precision: min cosine {cos:.5f}")
    if cos < 0.999:
        raise SystemExit("half-precision weights change the embeddings too much")
    # which run is behind the exported (and then deployed) model: the monthly retraining starts from it
    (args.out.parent / "served_run.txt").write_text(str(run.relative_to(EPILLS_ROOT)).replace("\\", "/"),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()
