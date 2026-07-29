#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/rl_common.sh"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointercad-rl-local}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_dataset.yaml}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HF_HOME:-}}"
PROXY_SCRIPT="${PROXY_SCRIPT:-}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/rl_data}"

rl_prepare_environment \
    "$REPO_ROOT" \
    "$CONDA_ENV_NAME" \
    "$HF_CACHE_DIR" \
    "$PROXY_SCRIPT"
rl_require_file "$CONFIG_PATH" "Episode builder config"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/build_episodes_$(rl_timestamp).log"

echo "[INFO] Conda environment: $CONDA_ENV_NAME"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Log: $LOG_PATH"

python -u -m preprocessing.build_rl_episode_index \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
