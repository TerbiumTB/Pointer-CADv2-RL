#!/usr/bin/env bash

set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pointercad
export HF_HOME="/mnt/afs_01e/mayi-folder/hf-cache"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

echo -e "${BLUE}=============================="
echo -e "   RL Training Launch Script"
echo -e "==============================${NC}"

# -----------------------------
# Test mode
# -----------------------------
if [[ "$@" == *"--test"* || "$@" == *"-t"* ]]; then
    echo -e "${BLUE}[DEBUG] Detected --test or -t flag, using local environment.${NC}"
    
    export MASTER_ADDR=localhost
    export MASTER_PORT=32501
    export WORLD_SIZE=$(nvidia-smi --list-gpus | wc -l)
    export NODE_RANK=0
    export GLOBAL_RANK_OFFSET=0

    echo -e "${GREEN}[INFO] MASTER_ADDR         = ${MASTER_ADDR}${NC}"
    echo -e "${GREEN}[INFO] MASTER_PORT         = ${MASTER_PORT}${NC}"
    echo -e "${GREEN}[INFO] WORLD_SIZE          = ${WORLD_SIZE}${NC}"
    echo -e "${GREEN}[INFO] NODE_RANK           = ${NODE_RANK}${NC}"
    echo -e "${GREEN}[INFO] GLOBAL_RANK_OFFSET  = ${GLOBAL_RANK_OFFSET}${NC}"

# -----------------------------
# Cluster mode
# -----------------------------
else
    ENV_SCRIPT=~/dist_env.sh
    echo -e "${YELLOW}[INFO] Loading distributed environment from ${ENV_SCRIPT}.${NC}"

    if [[ -f ${ENV_SCRIPT} ]]; then
        source ${ENV_SCRIPT}
        echo -e "${GREEN}[INFO] Successfully sourced ${ENV_SCRIPT}.${NC}"
    else
        echo -e "${RED}[WARN] Environment script ${ENV_SCRIPT} not found. Falling back to SenseCore variables.${NC}"

        # Compute environment vars from SenseCore-provided envs
        export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
        export MASTER_PORT=${MASTER_PORT:-32501}
        export WORLD_SIZE=$(( ${SENSECORE_PYTORCH_NNODES:-1} * ${SENSECORE_ACCELERATE_DEVICE_COUNT:-1} ))
        export NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-0}
        export GLOBAL_RANK_OFFSET=$(( ${SENSECORE_PYTORCH_NODE_RANK:-0} * ${SENSECORE_ACCELERATE_DEVICE_COUNT:-1} ))

        echo -e "${GREEN}[INFO] MASTER_ADDR         = ${MASTER_ADDR}${NC}"
        echo -e "${GREEN}[INFO] MASTER_PORT         = ${MASTER_PORT}${NC}"
        echo -e "${GREEN}[INFO] WORLD_SIZE          = ${WORLD_SIZE}${NC}"
        echo -e "${GREEN}[INFO] NODE_RANK           = ${NODE_RANK}${NC}"
        echo -e "${GREEN}[INFO] GLOBAL_RANK_OFFSET  = ${GLOBAL_RANK_OFFSET}${NC}"

        # Warn if key vars missing
        [[ -z "${SENSECORE_ACCELERATE_DEVICE_COUNT}" ]] && echo -e "${YELLOW}[WARN] SENSECORE_ACCELERATE_DEVICE_COUNT not set, defaulting to 1.${NC}"
        [[ -z "${SENSECORE_PYTORCH_NNODES}" ]] && echo -e "${YELLOW}[WARN] SENSECORE_PYTORCH_NNODES not set, defaulting to 1.${NC}"
        [[ -z "${SENSECORE_PYTORCH_NODE_RANK}" ]] && echo -e "${YELLOW}[WARN] SENSECORE_PYTORCH_NODE_RANK not set, defaulting to 0.${NC}"
    fi
fi

# -----------------------------
# Proxy setup
# -----------------------------
if [[ -f ~/proxy.sh ]]; then
    source ~/proxy.sh
    echo -e "${GREEN}[INFO] Proxy loaded from ~/proxy.sh${NC}"
else
    echo -e "${YELLOW}[WARN] No proxy.sh found, skipping proxy setup.${NC}"
fi

# -----------------------------
# Launch training
# -----------------------------
echo -e "${BLUE}Running python $REPO_ROOT/rl_train.py${NC}"
python -u "$REPO_ROOT/rl_train.py" 2>&1 | tee "$REPO_ROOT/rl_train.${NODE_RANK}.$(date +"%Y%m%d_%H%M%S").log"
