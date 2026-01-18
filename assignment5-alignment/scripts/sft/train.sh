#!/usr/bin/env bash
set -euo pipefail

# Run from the repo root regardless of where this script is invoked from
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"


TRAIN_CONFIG_JSON="./configs/sft/train_config.json"
# Dataset name can be 'math' 'gsm8k' 'mmlu' for now
DATASET_NAME="math"

uv run python train_sft.py \
  --train_config_json "$TRAIN_CONFIG_JSON" \
  --dataset_name "$DATASET_NAME"