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

POINTERCAD_CONDA_ENV="${POINTERCAD_CONDA_ENV:-pointercad}"
POINTERCAD_CONDA_EXE="${POINTERCAD_CONDA_EXE:-conda}"
if ! CONDA_HOOK="$("$POINTERCAD_CONDA_EXE" shell.bash hook)"; then
    echo "[ERROR] Cannot initialize conda. Check POINTERCAD_CONDA_EXE in .env." >&2
    exit 1
fi

eval "$CONDA_HOOK"
conda activate "$POINTERCAD_CONDA_ENV"

POINTERCAD_LOG_ROOT="${POINTERCAD_LOG_ROOT:-$REPO_ROOT/log}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/dpo_train.yaml}"
LOG_DIR="${LOG_DIR:-$POINTERCAD_LOG_ROOT/dpo_launch}"
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

echo -e "${BLUE}=============================="
echo -e "   DPO Training Launch Script"
echo -e "==============================${NC}"

TEST_MODE=false
for argument in "$@"; do
    case "$argument" in
        --test|-t) TEST_MODE=true ;;
    esac
done

if "$TEST_MODE"; then
    MASTER_ADDR="${MASTER_ADDR:-localhost}"
    MASTER_PORT="${MASTER_PORT:-32502}"
    NUM_MACHINES=1
    MACHINE_RANK=0
else
    POINTERCAD_DIST_ENV_SCRIPT="${POINTERCAD_DIST_ENV_SCRIPT:-$HOME/dist_env.sh}"
    echo -e "${YELLOW}[INFO] Loading distributed environment from ${POINTERCAD_DIST_ENV_SCRIPT}.${NC}"

    if [[ -f "$POINTERCAD_DIST_ENV_SCRIPT" ]]; then
        # shellcheck disable=SC1090
        source "$POINTERCAD_DIST_ENV_SCRIPT"
        echo -e "${GREEN}[INFO] Successfully sourced ${POINTERCAD_DIST_ENV_SCRIPT}.${NC}"
    else
        echo -e "${YELLOW}[WARN] Distributed environment script not found; using scheduler variables: ${POINTERCAD_DIST_ENV_SCRIPT}.${NC}"
    fi

    MASTER_ADDR="${MASTER_ADDR:-localhost}"
    MASTER_PORT="${MASTER_PORT:-32502}"
    NUM_MACHINES="${NUM_MACHINES:-${SENSECORE_PYTORCH_NNODES:-1}}"
    MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-0}}}"
fi

DETECTED_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "$DETECTED_GPUS" -le 0 ]]; then
    echo -e "${RED}[ERROR] PyTorch does not see any CUDA devices in $POINTERCAD_CONDA_ENV.${NC}"
    python -c "import torch; print('torch:', torch.__version__); print('compiled CUDA:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
    exit 1
fi

if "$TEST_MODE"; then
    GPUS_PER_NODE="$DETECTED_GPUS"
    NUM_PROCESSES="$DETECTED_GPUS"
else
    GPUS_PER_NODE="${GPUS_PER_NODE:-${NUM_GPUS:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-$DETECTED_GPUS}}}"
    NUM_PROCESSES="${NUM_PROCESSES:-$((NUM_MACHINES * GPUS_PER_NODE))}"
fi
if [[ "$GPUS_PER_NODE" -le 0 || "$GPUS_PER_NODE" -gt "$DETECTED_GPUS" ]]; then
    echo -e "${RED}[ERROR] GPUS_PER_NODE=$GPUS_PER_NODE, but PyTorch sees $DETECTED_GPUS CUDA devices.${NC}"
    exit 1
fi

WORLD_SIZE="$NUM_PROCESSES"
NODE_RANK="$MACHINE_RANK"
GLOBAL_RANK_OFFSET=$((MACHINE_RANK * GPUS_PER_NODE))
export MASTER_ADDR MASTER_PORT WORLD_SIZE NODE_RANK GLOBAL_RANK_OFFSET

echo -e "${GREEN}[INFO] MASTER_ADDR         = ${MASTER_ADDR}${NC}"
echo -e "${GREEN}[INFO] MASTER_PORT         = ${MASTER_PORT}${NC}"
echo -e "${GREEN}[INFO] WORLD_SIZE          = ${WORLD_SIZE}${NC}"
echo -e "${GREEN}[INFO] NODE_RANK           = ${NODE_RANK}${NC}"
echo -e "${GREEN}[INFO] GLOBAL_RANK_OFFSET  = ${GLOBAL_RANK_OFFSET}${NC}"
echo -e "${GREEN}[INFO] NUM_MACHINES        = ${NUM_MACHINES}${NC}"
echo -e "${GREEN}[INFO] MACHINE_RANK        = ${MACHINE_RANK}${NC}"
echo -e "${GREEN}[INFO] GPUS_PER_NODE       = ${GPUS_PER_NODE}${NC}"
echo -e "${GREEN}[INFO] NUM_PROCESSES       = ${NUM_PROCESSES}${NC}"
echo -e "${GREEN}[INFO] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-scheduler managed}${NC}"
echo -e "${GREEN}[INFO] torch CUDA devices   = ${DETECTED_GPUS}${NC}"

POINTERCAD_PROXY_ENV_SCRIPT="${POINTERCAD_PROXY_ENV_SCRIPT:-$HOME/proxy.sh}"
if [[ -f "$POINTERCAD_PROXY_ENV_SCRIPT" ]]; then
    # shellcheck disable=SC1090
    source "$POINTERCAD_PROXY_ENV_SCRIPT"
    echo -e "${GREEN}[INFO] Proxy loaded from $POINTERCAD_PROXY_ENV_SCRIPT${NC}"
else
    echo -e "${YELLOW}[WARN] Proxy environment script not found, skipping: $POINTERCAD_PROXY_ENV_SCRIPT${NC}"
fi

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/dpo_node_${MACHINE_RANK}_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $POINTERCAD_CONDA_ENV"
echo "[INFO] Python executable: $(python -c 'import sys; print(sys.executable)')"
echo "[INFO] HF_HOME: ${HF_HOME:-framework default}"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Accelerate processes: $NUM_PROCESSES"
echo "[INFO] Master port: $MASTER_PORT"
echo "[INFO] Log: $LOG_PATH"

accelerate launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines "$NUM_MACHINES" \
    --machine_rank "$MACHINE_RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    "$REPO_ROOT/dpo_train.py" \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
