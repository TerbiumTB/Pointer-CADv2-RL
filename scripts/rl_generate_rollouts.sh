#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointercad-rl-local}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_rollouts.yaml}"
GPU_ID="${GPU_ID:-0}"
HF_HOME="${HF_HOME:-/mnt/afs_01e/mayi-folder/hf-cache}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/rl_rollouts}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export HF_HOME
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/generate_rollouts_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $CONDA_ENV_NAME"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Physical GPU: $GPU_ID"
echo "[INFO] Log: $LOG_PATH"

python -u -m preprocessing.generate_rl_rollouts \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
