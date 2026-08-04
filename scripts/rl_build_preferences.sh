#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

POINTERCAD_ENV_FILE="${POINTERCAD_ENV_FILE:-$REPO_ROOT/.env}"
if [[ -f "$POINTERCAD_ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$POINTERCAD_ENV_FILE"
    set +a
fi

POINTERCAD_CONDA_ENV="${POINTERCAD_CONDA_ENV:-pointercad}"
POINTERCAD_CONDA_EXE="${POINTERCAD_CONDA_EXE:-conda}"
if ! CONDA_HOOK="$("$POINTERCAD_CONDA_EXE" shell.bash hook)"; then
    echo "[ERROR] Cannot initialize conda. Check POINTERCAD_CONDA_EXE in .env." >&2
    exit 1
fi

eval "$CONDA_HOOK"
conda activate "$POINTERCAD_CONDA_ENV"

POINTERCAD_LOG_ROOT="${POINTERCAD_LOG_ROOT:-$REPO_ROOT/log}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_preferences.yaml}"
LOG_DIR="${LOG_DIR:-$POINTERCAD_LOG_ROOT/rl_data}"

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/build_preferences_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $POINTERCAD_CONDA_ENV"
echo "[INFO] Python executable: $(python -c 'import sys; print(sys.executable)')"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Log: $LOG_PATH"

python -u -m preprocessing.build_rl_preferences \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
