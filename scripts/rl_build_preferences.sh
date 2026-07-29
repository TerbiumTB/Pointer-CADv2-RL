#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointercad-rl-local}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_preferences.yaml}"
HF_HOME="${HF_HOME:-/mnt/afs_01e/mayi-folder/hf-cache}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/rl_data}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"
export HF_HOME

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/build_preferences_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $CONDA_ENV_NAME"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Log: $LOG_PATH"

python -u -m preprocessing.build_rl_preferences \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
