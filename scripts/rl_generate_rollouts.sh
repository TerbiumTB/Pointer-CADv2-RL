#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pointercad-rl-local}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/rl_rollouts.yaml}"
HF_HOME="${HF_HOME:-/mnt/afs_01e/mayi-folder/hf-cache}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/log/rl_rollouts}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

export HF_HOME
export PYTORCH_CUDA_ALLOC_CONF

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================"
echo -e "   RL Rollout Generation"
echo -e "================================${NC}"

if [[ "$@" == *"--test"* || "$@" == *"-t"* ]]; then
    export MASTER_ADDR=localhost
    export MASTER_PORT=32501
    export WORLD_SIZE=$(nvidia-smi --list-gpus | wc -l)
    export NODE_RANK=0
    export GLOBAL_RANK_OFFSET=0
else
    ENV_SCRIPT=~/dist_env.sh
    echo -e "${YELLOW}[INFO] Loading distributed environment from ${ENV_SCRIPT}.${NC}"

    if [[ -f ${ENV_SCRIPT} ]]; then
        source ${ENV_SCRIPT}
        echo -e "${GREEN}[INFO] Successfully sourced ${ENV_SCRIPT}.${NC}"
    else
        echo -e "${RED}[WARN] Environment script ${ENV_SCRIPT} not found. Falling back to SenseCore variables.${NC}"

        export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
        export MASTER_PORT=${MASTER_PORT:-32501}
        export WORLD_SIZE=$(( ${SENSECORE_PYTORCH_NNODES:-1} * ${SENSECORE_ACCELERATE_DEVICE_COUNT:-1} ))
        export NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-0}
        export GLOBAL_RANK_OFFSET=$(( ${SENSECORE_PYTORCH_NODE_RANK:-0} * ${SENSECORE_ACCELERATE_DEVICE_COUNT:-1} ))
    fi
fi

CUDA_DEVICE_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "$CUDA_DEVICE_COUNT" -le 0 ]]; then
    echo -e "${RED}[ERROR] PyTorch does not see any CUDA devices in $CONDA_ENV_NAME.${NC}"
    python -c "import torch; print('torch:', torch.__version__); print('compiled CUDA:', torch.version.cuda); print('cuda available:', torch.cuda.is_available())"
    exit 1
fi

echo -e "${GREEN}[INFO] MASTER_ADDR         = ${MASTER_ADDR}${NC}"
echo -e "${GREEN}[INFO] MASTER_PORT         = ${MASTER_PORT}${NC}"
echo -e "${GREEN}[INFO] WORLD_SIZE          = ${WORLD_SIZE}${NC}"
echo -e "${GREEN}[INFO] NODE_RANK           = ${NODE_RANK}${NC}"
echo -e "${GREEN}[INFO] GLOBAL_RANK_OFFSET  = ${GLOBAL_RANK_OFFSET}${NC}"
echo -e "${GREEN}[INFO] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-scheduler managed}${NC}"
echo -e "${GREEN}[INFO] torch CUDA devices   = ${CUDA_DEVICE_COUNT}${NC}"

if [[ -f ~/proxy.sh ]]; then
    source ~/proxy.sh
    echo -e "${GREEN}[INFO] Proxy loaded from ~/proxy.sh${NC}"
else
    echo -e "${YELLOW}[WARN] No proxy.sh found, skipping proxy setup.${NC}"
fi

mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/generate_rollouts_$(date +"%Y%m%d_%H%M%S").log"

echo "[INFO] Conda environment: $CONDA_ENV_NAME"
echo "[INFO] Config: $CONFIG_PATH"
echo "[INFO] Log: $LOG_PATH"

python -u -m preprocessing.generate_rl_rollouts \
    -c "$CONFIG_PATH" \
    2>&1 | tee "$LOG_PATH"
