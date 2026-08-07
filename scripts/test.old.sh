#!/usr/bin/env bash

# Legacy HTTP-coordinated evaluation launcher retained for reproducibility.

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

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/test.old.yaml}"
PORT="${POINTERCAD_EVAL_PORT:-32500}"
WORKERS_PER_GPU="${POINTERCAD_EVAL_WORKERS_PER_GPU:-1}"
OUTPUT_DIR="${POINTERCAD_EVAL_OUTPUT_DIR:-}"

usage() {
    cat <<EOF
Usage: scripts/test.old.sh [options]

Run SFT progressive CAD evaluation on all visible GPUs.

Options:
  -c, --config PATH             Evaluation YAML (default: config/test.old.yaml)
  -p, --port PORT               Local coordinator port (default: 32500)
  -w, --workers-per-gpu N       Workers per visible GPU (default: 1)
  -o, --output-dir PATH         Exact run output directory
      --help                    Show this help

The same values can be set with CONFIG_PATH, POINTERCAD_EVAL_PORT,
POINTERCAD_EVAL_WORKERS_PER_GPU and POINTERCAD_EVAL_OUTPUT_DIR.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers-per-gpu)
            WORKERS_PER_GPU="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
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
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "[ERROR] Port must be an integer between 1 and 65535: $PORT" >&2
    exit 1
fi
if [[ ! "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Workers per GPU must be a positive integer: $WORKERS_PER_GPU" >&2
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
export PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM

POINTERCAD_PROXY_ENV_SCRIPT="${POINTERCAD_PROXY_ENV_SCRIPT:-$HOME/proxy.sh}"
if [[ -f "$POINTERCAD_PROXY_ENV_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$POINTERCAD_PROXY_ENV_SCRIPT"
fi
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1"
export NO_PROXY="$no_proxy"

DETECTED_GPUS="$(python -c "import torch; print(torch.cuda.device_count())")"
if [[ "$DETECTED_GPUS" -le 0 ]]; then
    echo "[ERROR] PyTorch does not see any CUDA devices in $POINTERCAD_CONDA_ENV." >&2
    exit 1
fi

GPU_IDS=()
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
fi
if [[ "${#GPU_IDS[@]}" -ne "$DETECTED_GPUS" ]]; then
    GPU_IDS=()
    for ((gpu_index=0; gpu_index<DETECTED_GPUS; gpu_index++)); do
        GPU_IDS+=("$gpu_index")
    done
fi

TOTAL_WORKERS=$((DETECTED_GPUS * WORKERS_PER_GPU))
POINTERCAD_LOG_ROOT="${POINTERCAD_LOG_ROOT:-$REPO_ROOT/log}"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$POINTERCAD_LOG_ROOT/sft_eval/$(date +"%Y-%m-%d/%H-%M-%S")"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
CONFIG_PATH="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"

if ! python -c 'import socket, sys; s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.close()' "$PORT"; then
    echo "[ERROR] Coordinator port is already in use: $PORT" >&2
    exit 1
fi

SERVER_PID=""
WORKER_PIDS=()
cleanup() {
    for pid in "${WORKER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Conda environment: $POINTERCAD_CONDA_ENV"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Visible GPUs: $DETECTED_GPUS"
echo "[INFO] Workers: $TOTAL_WORKERS ($WORKERS_PER_GPU per GPU)"
echo "[INFO] Output: $OUTPUT_DIR"

python -u "$REPO_ROOT/test_server.py" \
    -p "$PORT" \
    --expected-clients "$TOTAL_WORKERS" \
    > >(tee "$OUTPUT_DIR/server.log") 2>&1 &
SERVER_PID=$!

SERVER_READY=false
for ((attempt=0; attempt<100; attempt++)); do
    if python -c 'import socket, sys; s = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=0.2); s.close()' "$PORT" 2>/dev/null; then
        SERVER_READY=true
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        break
    fi
    sleep 0.2
done
if [[ "$SERVER_READY" != true ]]; then
    echo "[ERROR] Evaluation coordinator did not start. See $OUTPUT_DIR/server.log" >&2
    exit 1
fi

worker_index=0
for gpu_id in "${GPU_IDS[@]}"; do
    for ((local_worker=0; local_worker<WORKERS_PER_GPU; local_worker++)); do
        worker_log="$OUTPUT_DIR/worker_${worker_index}.log"
        echo "[INFO] Starting worker $worker_index on GPU $gpu_id"
        CUDA_VISIBLE_DEVICES="$gpu_id" python -u "$REPO_ROOT/test.old.py" \
            -c "$CONFIG_PATH" \
            -h 127.0.0.1 \
            -p "$PORT" \
            -o "$OUTPUT_DIR" \
            > "$worker_log" 2>&1 &
        WORKER_PIDS+=("$!")
        worker_index=$((worker_index + 1))
    done
done

WORKER_FAILURE=false
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then
        WORKER_FAILURE=true
    fi
done
if [[ "$WORKER_FAILURE" == true ]]; then
    echo "[ERROR] At least one evaluation worker failed. See $OUTPUT_DIR/worker_*.log" >&2
    exit 1
fi

if ! wait "$SERVER_PID"; then
    echo "[ERROR] Evaluation coordinator failed. See $OUTPUT_DIR/server.log" >&2
    exit 1
fi
SERVER_PID=""

python -u "$REPO_ROOT/eval.py" \
    -i "$OUTPUT_DIR/results.json" \
    2>&1 | tee "$OUTPUT_DIR/eval.log"

echo "[INFO] Evaluation finished: $OUTPUT_DIR"
