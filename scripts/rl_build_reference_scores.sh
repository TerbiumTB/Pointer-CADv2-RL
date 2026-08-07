#!/usr/bin/env bash

set -eo pipefail

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

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_reference_scores.yaml}"

usage() {
    cat <<EOF
Usage: scripts/rl_build_reference_scores.sh [options]

Options:
  -c, --config PATH  Score builder YAML (default: config/rl_reference_scores.yaml)
      --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            if [[ $# -lt 2 || -z "$2" ]]; then
                echo "[ERROR] $1 requires a config path." >&2
                exit 2
            fi
            CONFIG_PATH="$2"
            shift 2
            ;;
        --config=*)
            CONFIG_PATH="${1#*=}"
            if [[ -z "$CONFIG_PATH" ]]; then
                echo "[ERROR] --config requires a config path." >&2
                exit 2
            fi
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Reference score config not found: $CONFIG_PATH" >&2
    exit 1
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
LOG_DIR="${LOG_DIR:-$POINTERCAD_LOG_ROOT/rl_data}"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/build_reference_scores_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $POINTERCAD_CONDA_ENV"
echo "[INFO] Python executable: $(python -c 'import sys; print(sys.executable)')"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Log: $LOG_PATH"

python -u -m preprocessing.build_rl_reference_scores \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
