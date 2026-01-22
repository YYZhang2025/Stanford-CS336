#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"  
cd "$REPO_ROOT"


TRAIN_CONFIG_JSON="./configs/ei/train_config.json"
# Dataset name can be 'math' 'gsm8k' for now
DATASET_NAME="gsm8k"

uv run python train_ei.py \
  --train_config_path "$TRAIN_CONFIG_JSON" \
  --dataset_name "$DATASET_NAME"