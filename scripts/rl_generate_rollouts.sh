#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/rl_common.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointercad-rl-local}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_rollouts.yaml}"
GPU_ID="${GPU_ID:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-}}"
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/rl_rollouts}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

rl_prepare_environment \
    "$REPO_ROOT" \
    "$CONDA_ENV_NAME" \
    "$HF_CACHE_DIR" \
    "$PROXY_SCRIPT"
rl_require_file "$CONFIG_PATH" "Rollout config"
mkdir -p "$LOG_DIR"

if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_ID must contain one physical GPU index." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF
LOG_PATH="$LOG_DIR/generate_rollouts_$(rl_timestamp).log"

echo "[INFO] Conda environment: $CONDA_ENV_NAME"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Physical GPU: $GPU_ID"
echo "[INFO] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[INFO] Current rollout generator uses one GPU per process."
echo "[INFO] Log: $LOG_PATH"

python -u -m preprocessing.generate_rl_rollouts \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
