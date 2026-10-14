#!/usr/bin/env bash
# Unattended evaluation of a trained imprint-reader adapter against the served reader (large_c92), run after training:
#   1. keep the epoch whose validation identify (best re-ranking weight) is highest -> adapter_best.pt
#   2. re-read every validation, held-out and ePillID crop with it (scripts/read_imprints_cache.py --preset <name>)
#   3. choose identify's re-ranking on validation and test it once (scripts/tune_identify_rerank.py)
#   4. the gallery imprint check, chosen on validation, tested once (scripts/evaluate_imprint_gallery.py)
# Usage, from the epills folder (Git Bash):  bash scripts/evaluate_reader_candidate.sh large_lora_v2
set -euo pipefail
NAME="$1"
DIR="outputs/imprint_reader/$NAME"
LOG="outputs/logs/train_imprint_reader_${NAME#large_lora_}.log"
PY=.venv/Scripts/python
until grep -qE "^wrote|Traceback" "$LOG"; do sleep 60; done
grep -q Traceback "$LOG" && { echo "training failed: $LOG"; exit 1; }
"$PY" - "$DIR" <<'EOF'
import json, shutil, sys
from pathlib import Path
d = Path(sys.argv[1]); ev = json.loads((d / "eval.json").read_text())
def best(r): return max(v for k, v in r["adapter"].items() if k.startswith("identify_top1_alpha_") and k != "identify_top1_alpha_0")
epochs = sorted(k for k in ev if k.startswith("epoch_"))
pick = max(epochs, key=lambda e: best(ev[e]))
shutil.copy(d / f"adapter_e{pick.split('_')[1]}.pt", d / "adapter_best.pt")
print("kept", pick, {e: round(best(ev[e]), 3) for e in epochs})
EOF
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u scripts/read_imprints_cache.py --preset "$NAME"
"$PY" -u scripts/tune_identify_rerank.py --run outputs/runs/stage2_run4 --ocr "$NAME"
"$PY" -u scripts/evaluate_imprint_gallery.py --run outputs/runs/stage2_run4 --ocr "$NAME"
echo "done: $NAME"
