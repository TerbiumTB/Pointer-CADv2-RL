#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/rl_common.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointercad-rl-local}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/dpo_train.yaml}"
GPU_IDS="${GPU_IDS:-0}"
NUM_GPUS="${NUM_GPUS:-$(rl_gpu_count "$GPU_IDS")}"
MASTER_PORT="${MASTER_PORT:-29501}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-}}"
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/dpo_launch}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

rl_prepare_environment \
    "$REPO_ROOT" \
    "$CONDA_ENV_NAME" \
    "$HF_CACHE_DIR" \
    "$PROXY_SCRIPT"
rl_require_file "$CONFIG_PATH" "DPO config"
mkdir -p "$LOG_DIR"

if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "[ERROR] GPU_IDS must look like 0 or 0,1,2,3." >&2
    exit 1
fi
if [[ ! "$NUM_GPUS" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] NUM_GPUS must be an integer." >&2
    exit 1
fi

VISIBLE_GPU_COUNT="$(rl_gpu_count "$GPU_IDS")"
if (( NUM_GPUS <= 0 )); then
    echo "[ERROR] NUM_GPUS must be positive for DPO training." >&2
    exit 1
fi
if (( NUM_GPUS > VISIBLE_GPU_COUNT )); then
    echo "[ERROR] NUM_GPUS=$NUM_GPUS exceeds visible GPU count $VISIBLE_GPU_COUNT." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF
export NCCL_DEBUG

LAUNCH_ARGS=(
    --num_processes "$NUM_GPUS"
    --num_machines 1
    --main_process_port "$MASTER_PORT"
)
if [[ -n "$ACCELERATE_CONFIG_FILE" ]]; then
    rl_require_file "$ACCELERATE_CONFIG_FILE" "Accelerate config"
    LAUNCH_ARGS=(--config_file "$ACCELERATE_CONFIG_FILE" "${LAUNCH_ARGS[@]}")
fi

LOG_PATH="$LOG_DIR/dpo_$(rl_timestamp).log"

echo "[INFO] Conda environment: $CONDA_ENV_NAME"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] GPU IDs: $GPU_IDS"
echo "[INFO] Accelerate processes: $NUM_GPUS"
echo "[INFO] Master port: $MASTER_PORT"
echo "[INFO] Log: $LOG_PATH"

accelerate launch \
    "${LAUNCH_ARGS[@]}" \
    "$REPO_ROOT/dpo_train.py" \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
