"""
Train the Stage 2 model (epills.core.models.hybrid.HybridPillNet) on the C3PI split.

Each epoch is evaluated on the validation splits (training/eval/retrieval.py); the best epoch by `select_metric` is
kept, as in the FER notebook's EarlyStopping(restore_best_weights=True). Outputs in outputs/runs/<name>/:
  best.pt        model weights + config + vocab + metrics of the best epoch (what serving loads)
  history.csv    one row per epoch: phase, losses, learning rates, validation metrics, time
  summary.json   the best epoch's metrics

Usage, from the epills folder:
  .venv\\Scripts\\python scripts\\train_stage2.py --config configs\\stage2_baseline.yaml
Smoke tests:
  --overfit 10       train and evaluate on 10 training groups only (should approach 100% top-1)
  --max-batches 20   cap batches per epoch;  --epochs 2   cap epochs per phase;  --phases 1   run only phase 1
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

EPILLS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EPILLS_ROOT / "src"))
from epills.core.models.hybrid import HybridPillNet  # noqa: E402
from epills.training.datasets.pills import (PillCrops, PKSampler, Vocab, eval_transform, load_manifest,  # noqa: E402
                                            train_transform)
from epills.training.eval.retrieval import embed, evaluate  # noqa: E402
from epills.training.losses.metric import ArcFace, auxiliary_losses, batch_hard_triplet  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model, metric_module, phase_cfg, weight_decay):
    """
    lr: pretrained backbones; head_lr: the new layers; metric_lr (default head_lr): the ArcFace class table, which
    starts random for 1,710 classes and barely moves at the FER rate.
    """
    backbone, new = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (backbone if name.startswith(("vgg.", "vit.")) else new).append(p)
    groups = [{"params": new, "lr": phase_cfg["head_lr"], "name": "new"}]
    metric = [p for p in metric_module.parameters() if p.requires_grad]
    if metric:
        groups.append({"params": metric, "lr": phase_cfg.get("metric_lr", phase_cfg["head_lr"]), "name": "metric"})
    if backbone:
        groups.append({"params": backbone, "lr": phase_cfg["lr"], "name": "backbone"})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", help="run name (default: the config's name)")
    parser.add_argument("--overfit", type=int, default=0, help="use only this many training groups for train and eval")
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=0, help="cap epochs per phase")
    parser.add_argument("--phases", type=int, nargs="+", help="run only these phases (smoke tests)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    name = args.name or cfg["name"]
    run_dir = EPILLS_ROOT / "outputs" / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    manifest = load_manifest(EPILLS_ROOT / cfg["data"]["manifest"])
    images = EPILLS_ROOT / cfg["data"]["images"]
    train_df = manifest[manifest.split == "train"]
    vocab = Vocab(train_df)
    if args.overfit:
        keep = sorted(train_df.label.unique())[:args.overfit]
        train_df = train_df[train_df.label.isin(keep)]
        known_df, unknown_df = train_df, manifest[manifest.split == "val_unknown"].iloc[:500]
    else:
        known_df, unknown_df = manifest[manifest.split == "val_enrolled"], manifest[manifest.split == "val_unknown"]

    train_set = PillCrops(train_df, images, train_transform(cfg["augment"]["colour_jitter"]), vocab)
    eval_tf = eval_transform()
    known_set, unknown_set = PillCrops(known_df, images, eval_tf, vocab), PillCrops(unknown_df, images, eval_tf, vocab)
    workers = cfg["data"]["workers"]
    loader_kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    known_loader = DataLoader(known_set, batch_size=cfg["eval"]["batch"], **loader_kw)
    unknown_loader = DataLoader(unknown_set, batch_size=cfg["eval"]["batch"], **loader_kw)
    print(f"train: {len(train_set):,} photos, {len(train_set.label_index):,} groups | eval known: {len(known_set):,} "
          f"photos, {known_df.label.nunique()} groups | unknown: {len(unknown_set):,} photos | device {device}", flush=True)

    mcfg, lcfg = cfg["model"], cfg["loss"]
    model = HybridPillNet(embedding_dim=mcfg["embedding_dim"], refine_pool=mcfg["refine_pool"],
                          num_shapes=len(vocab.shapes) if mcfg["aux_heads"] else 0,
                          num_colours=len(vocab.colours) if mcfg["aux_heads"] else 0).to(device)
    if lcfg["metric"] == "arcface":
        metric_module = ArcFace(mcfg["embedding_dim"], len(train_set.label_index), lcfg["arcface_margin"],
                                lcfg["arcface_scale"]).to(device)
        metric_loss = metric_module
    else:
        metric_module = torch.nn.Module()
        metric_loss = lambda emb, lab: batch_hard_triplet(emb, lab, lcfg["triplet_margin"])  # noqa: E731

    # ArcFace margin warm-up: the margin grows from 0 over the first epochs. With the full margin from the start, the
    # quickest way to lower the loss is to squeeze every embedding onto a few dimensions (the baseline collapsed to ~3).
    warmup = lcfg.get("arcface_margin_warmup_epochs", 0)
    history, best, global_epoch = [], {"score": -1.0}, 0
    select = cfg["select_metric"]
    for phase_cfg in cfg["phases"]:
        if args.phases and phase_cfg["phase"] not in args.phases:
            continue
        model.set_phase(phase_cfg["phase"])
        optimizer = build_optimizer(model, metric_module, phase_cfg, cfg["weight_decay"])
        plateau = cfg["plateau"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=plateau["factor"],
                                                               patience=plateau["patience"], min_lr=plateau["min_lr"])
        sampler = PKSampler(train_set.labels, p=phase_cfg["p"], k=phase_cfg["k"], seed=cfg["seed"] + phase_cfg["phase"])
        train_loader = DataLoader(train_set, batch_sampler=sampler, **loader_kw)
        epochs = min(phase_cfg["epochs"], args.epochs) if args.epochs else phase_cfg["epochs"]
        phase_best, wait = -1.0, 0
        print(f"\n=== Phase {phase_cfg['phase']}: up to {epochs} epochs, batch {phase_cfg['p']}x{phase_cfg['k']} ===", flush=True)
        for epoch in range(1, epochs + 1):
            global_epoch += 1
            if isinstance(metric_module, ArcFace):
                metric_module.margin = lcfg["arcface_margin"] * min(1.0, (global_epoch - 1) / warmup if warmup else 1.0)
            model.train()
            metric_module.train()
            t0, sums, n = time.time(), {}, 0
            for b, batch in enumerate(train_loader):
                if args.max_batches and b >= args.max_batches:
                    break
                image = batch["image"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(image)
                emb = out["embedding"].float()
                losses = {"metric": metric_loss(emb, label)}
                losses.update(auxiliary_losses(out, batch["shape"].to(device), batch["colour"].to(device)))
                total = (losses["metric"] + lcfg["lambda_shape"] * losses.get("shape", 0.0)
                         + lcfg["lambda_colour"] * losses.get("colour", 0.0))
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                for k, v in losses.items():
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
                n += 1
            train_time = time.time() - t0
            known = embed(model, known_loader, device)
            unknown = embed(model, unknown_loader, device)[0]
            metrics = evaluate(known[0], known[1], unknown, k_enrol=cfg["eval"]["k_enrol"], seed=cfg["seed"])
            score = metrics[select]
            scheduler.step(score)
            row = dict(phase=phase_cfg["phase"], epoch=epoch, margin=getattr(metric_module, "margin", None),
                       **{f"loss_{k}": v / max(n, 1) for k, v in sums.items()},
                       **{f"lr_{g['name']}": g["lr"] for g in optimizer.param_groups}, **metrics,
                       train_seconds=round(train_time, 1), eval_seconds=round(time.time() - t0 - train_time, 1))
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
            print(f"phase {phase_cfg['phase']} epoch {epoch:2d}: loss {row.get('loss_metric', 0):.3f} | top1 all "
                  f"{metrics['top1_all']:.3f} | gallery-10: top1 {metrics['top1_gallery10']:.3f}, unknown AUROC "
                  f"{metrics['auroc_gallery10']:.3f}, correct@95%-reject {metrics['known_correct_95_gallery10']:.3f} "
                  f"| dims {metrics['embedding_dims']:.1f} | {train_time:.0f}+{row['eval_seconds']:.0f}s", flush=True)
            if score > best["score"]:
                best = {"score": score, "phase": phase_cfg["phase"], "epoch": epoch, "metrics": metrics,
                        "state": copy.deepcopy(model.state_dict()), "metric_state": copy.deepcopy(metric_module.state_dict())}
                torch.save({"model": best["state"], "config": cfg, "vocab": vocab.to_dict(), "metrics": metrics,
                            "phase": best["phase"], "epoch": epoch}, run_dir / "best.pt")
            if score > phase_best:
                phase_best, wait = score, 0
            else:
                wait += 1
                if wait >= cfg["early_stopping_patience"]:
                    print(f"early stop: no gain in {select} for {wait} epochs", flush=True)
                    break
        # restore the best weights before the next phase (FER: restore_best_weights=True)
        model.load_state_dict(best["state"])
        metric_module.load_state_dict(best["metric_state"])

    summary = {"run": name, "best_phase": best["phase"], "best_epoch": best["epoch"], select: best["score"],
               **best["metrics"]}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nBest:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
