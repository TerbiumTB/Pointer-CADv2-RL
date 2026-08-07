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

CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_rollouts.yaml}"
FORWARD_ARGS=()

usage() {
    cat <<EOF
Usage: scripts/rl_generate_rollouts.sh [options]

Options:
  -c, --config PATH             Rollout YAML (default: config/rl_rollouts.yaml)
  -f, --force                   Recreate the target rollout run
      --batch-size N            Override generation.batch_size
      --cpu-workers-per-gpu N   Override generation.cpu_workers_per_gpu
      --batch-wait-seconds N    Override generation.batch_wait_seconds
      --help                    Show this help
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
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Rollout config not found: $CONFIG_PATH" >&2
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
LOG_DIR="${LOG_DIR:-$POINTERCAD_LOG_ROOT/rl_rollouts}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export PYTORCH_CUDA_ALLOC_CONF TOKENIZERS_PARALLELISM
if [[ -n "${HF_HOME:-}" ]]; then
    export HF_HOME
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================"
echo -e "   RL Rollout Generation"
echo -e "================================${NC}"

CUDA_DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "$CUDA_DEVICE_COUNT" -le 0 ]]; then
    echo -e "${RED}[ERROR] PyTorch does not see any CUDA devices in $POINTERCAD_CONDA_ENV.${NC}"
    python -c "import torch; print('torch:', torch.__version__); print('compiled CUDA:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
    exit 1
fi

echo -e "${GREEN}[INFO] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-scheduler managed}${NC}"
echo -e "${GREEN}[INFO] torch CUDA devices   = ${CUDA_DEVICE_COUNT}${NC}"

POINTERCAD_DIST_ENV_SCRIPT="${POINTERCAD_DIST_ENV_SCRIPT:-$HOME/dist_env.sh}"
if [[ -f "$POINTERCAD_DIST_ENV_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$POINTERCAD_DIST_ENV_SCRIPT"
    echo -e "${GREEN}[INFO] Distributed environment loaded from $POINTERCAD_DIST_ENV_SCRIPT${NC}"
fi

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-32503}"
NUM_MACHINES="${NUM_MACHINES:-${SENSECORE_PYTORCH_NNODES:-1}}"
MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-0}}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${NUM_GPUS:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-$CUDA_DEVICE_COUNT}}}"
if [[ "$GPUS_PER_NODE" -le 0 || "$GPUS_PER_NODE" -gt "$CUDA_DEVICE_COUNT" ]]; then
    echo -e "${RED}[ERROR] GPUS_PER_NODE=$GPUS_PER_NODE, but PyTorch sees $CUDA_DEVICE_COUNT CUDA devices.${NC}"
    exit 1
fi

echo -e "${GREEN}[INFO] NUM_MACHINES        = ${NUM_MACHINES}${NC}"
echo -e "${GREEN}[INFO] MACHINE_RANK        = ${MACHINE_RANK}${NC}"
echo -e "${GREEN}[INFO] GPUS_PER_NODE       = ${GPUS_PER_NODE}${NC}"
echo -e "${GREEN}[INFO] MASTER_ADDR         = ${MASTER_ADDR}${NC}"
echo -e "${GREEN}[INFO] MASTER_PORT         = ${MASTER_PORT}${NC}"

POINTERCAD_PROXY_ENV_SCRIPT="${POINTERCAD_PROXY_ENV_SCRIPT:-$HOME/proxy.sh}"
if [[ -f "$POINTERCAD_PROXY_ENV_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$POINTERCAD_PROXY_ENV_SCRIPT"
    echo -e "${GREEN}[INFO] Proxy loaded from $POINTERCAD_PROXY_ENV_SCRIPT${NC}"
else
    echo -e "${YELLOW}[WARN] Proxy environment script not found, skipping: $POINTERCAD_PROXY_ENV_SCRIPT${NC}"
fi

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/generate_rollouts_node_${MACHINE_RANK}_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $POINTERCAD_CONDA_ENV"
echo "[INFO] Python executable: $(python -c 'import sys; print(sys.executable)')"
echo "[INFO] HF_HOME: ${HF_HOME:-framework default}"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Log: $LOG_PATH"

torchrun \
    --nnodes "$NUM_MACHINES" \
    --nproc-per-node "$GPUS_PER_NODE" \
    --node-rank "$MACHINE_RANK" \
    --master-addr "$MASTER_ADDR" \
    --master-port "$MASTER_PORT" \
    -m preprocessing.generate_rl_rollouts \
    -c "$CONFIG_PATH" \
    "${FORWARD_ARGS[@]}" \
    2>&1 | tee "$LOG_PATH"
