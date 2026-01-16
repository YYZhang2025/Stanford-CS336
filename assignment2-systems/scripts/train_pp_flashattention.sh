#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_CONFIG_JSON="./configs/ddp_flash_attn/train_config.json"
MODEL_CONFIG_JSON="./configs/ddp_flash_attn/model_config.json"

# ------------------------------------------------------------
# Auto-detect number of processes = number of visible GPUs.
# - If CUDA_VISIBLE_DEVICES is set, respect it.
# - Else try nvidia-smi.
# - Else try torch.cuda.device_count().
# - Fallback: 1
# ------------------------------------------------------------
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Count comma-separated entries
  NPROC="$(python - <<'PY'
import os
s = os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
if not s:
    print(1)
else:
    ids = [x for x in s.split(",") if x.strip()!=""]
    print(max(1, len(ids)))
PY
)"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
    [[ -z "$NPROC" || "$NPROC" -le 0 ]] && NPROC=1
  else
    NPROC="$(uv run python - <<'PY'
import torch
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(max(1, n))
PY
)"
  fi
fi

echo "[info] Using nproc_per_node=${NPROC}"

uv run torchrun --standalone --nproc_per_node="${NPROC}" train_parallel.py \
  --train_config_json "$TRAIN_CONFIG_JSON" \
  --model_config_json "$MODEL_CONFIG_JSON"