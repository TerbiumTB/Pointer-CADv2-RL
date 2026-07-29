#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointercad-rl-local}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/dpo_train.yaml}"
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
HF_HOME="${HF_HOME:-/mnt/afs_01e/mayi-folder/hf-cache}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/dpo_launch}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

export HF_HOME
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/dpo_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $CONDA_ENV_NAME"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] GPU IDs: $GPU_IDS"
echo "[INFO] Accelerate processes: $NUM_GPUS"
echo "[INFO] Master port: $MASTER_PORT"
echo "[INFO] Log: $LOG_PATH"

accelerate launch \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 \
    --main_process_port "$MASTER_PORT" \
    "$REPO_ROOT/dpo_train.py" \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
