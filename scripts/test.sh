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

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/test.yaml}"
OUTPUT_DIR="${POINTERCAD_EVAL_OUTPUT_DIR:-}"
MASTER_PORT="${POINTERCAD_EVAL_PORT:-32500}"

usage() {
    cat <<EOF
Usage: scripts/test.sh [options]

Run deterministic full-episode CAD evaluation on all visible GPUs.

Options:
  -c, --config PATH       Evaluation YAML (default: config/test.yaml)
  -o, --output-dir PATH   Exact run output directory; an interrupted matching
                          run resumes from rank-safe JSONL shards
  -p, --port PORT         torchrun rendezvous port (default: 32500)
      --help              Show this help

GPU batch size and CPU workers per GPU are configured under generation in YAML.
The legacy HTTP-coordinated evaluator is available as scripts/test.old.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
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
    echo "[ERROR] Evaluation config not found: $CONFIG_PATH" >&2
    exit 1
fi
if [[ ! "$MASTER_PORT" =~ ^[0-9]+$ ]] || (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
    echo "[ERROR] Port must be an integer between 1 and 65535: $MASTER_PORT" >&2
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

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS="${POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS:-21600}"
if [[ ! "$POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM
export POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS

POINTERCAD_PROXY_ENV_SCRIPT="${POINTERCAD_PROXY_ENV_SCRIPT:-$HOME/proxy.sh}"
if [[ -f "$POINTERCAD_PROXY_ENV_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$POINTERCAD_PROXY_ENV_SCRIPT"
fi

DETECTED_GPUS="$(python -c "import torch; print(torch.cuda.device_count())")"
if [[ "$DETECTED_GPUS" -le 0 ]]; then
    echo "[ERROR] PyTorch does not see any CUDA devices in $POINTERCAD_CONDA_ENV." >&2
    exit 1
fi

POINTERCAD_LOG_ROOT="${POINTERCAD_LOG_ROOT:-$REPO_ROOT/log}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$POINTERCAD_LOG_ROOT/model_eval/$(date +"%Y-%m-%d/%H-%M-%S")"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

if ! python -c 'import socket, sys; s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$MASTER_PORT"; then
    echo "[ERROR] torchrun rendezvous port is already in use: $MASTER_PORT" >&2
    exit 1
fi

echo "[INFO] Conda environment: $POINTERCAD_CONDA_ENV"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Visible GPUs: $DETECTED_GPUS"
echo "[INFO] GPU coordinators: $DETECTED_GPUS (one per GPU)"
echo "[INFO] Distributed synchronization timeout: ${POINTERCAD_EVAL_DISTRIBUTED_TIMEOUT_SECONDS}s"
echo "[INFO] Output: $OUTPUT_DIR"

torchrun \
    --standalone \
    --nnodes 1 \
    --nproc-per-node "$DETECTED_GPUS" \
    --master-port "$MASTER_PORT" \
    "$REPO_ROOT/test.py" \
    -c "$CONFIG_PATH" \
    -o "$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

python -u "$REPO_ROOT/eval.py" \
    -i "$OUTPUT_DIR/results.json" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo "[INFO] Evaluation finished: $OUTPUT_DIR"
