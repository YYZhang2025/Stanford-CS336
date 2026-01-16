#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Run from the repo root regardless of where this script is invoked from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_CONFIG_JSON="./configs/ddp_flash_attn/train_config.json"
MODEL_CONFIG_JSON="./configs/ddp_flash_attn/model_config.json"

uv run torchrun --nproc_per_node=2 train_parallel.py \
  --train_config_json "$TRAIN_CONFIG_JSON" \
  --model_config_json "$MODEL_CONFIG_JSON"