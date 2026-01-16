#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_CONFIG_JSON="./configs/triton_flash_attn/train_config.json"
MODEL_CONFIG_JSON="./configs/triton_flash_attn/model_config.json"

uv run python train.py \
  --train_config_json "$TRAIN_CONFIG_JSON" \
  --model_config_json "$MODEL_CONFIG_JSON"