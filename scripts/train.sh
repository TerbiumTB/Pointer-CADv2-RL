#!/usr/bin/env bash

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

if [[ "$@" == *"--test"* || "$@" == *"-t"* ]]; then
    echo -e "${BLUE}[DEBUG] Detected --test or -t flag, setting custom environment variables.${NC}"
    
    export MASTER_ADDR=localhost
    export MASTER_PORT=32501
    export WORLD_SIZE=$(nvidia-smi --list-gpus | wc -l)
    export NODE_RANK=0
    export GLOBAL_RANK_OFFSET=0

    echo -e "${GREEN}[INFO] MASTER_ADDR set to: ${MASTER_ADDR}${NC}"
    echo -e "${GREEN}[INFO] MASTER_PORT set to: ${MASTER_PORT}${NC}"
    echo -e "${GREEN}[INFO] WORLD_SIZE set to: ${WORLD_SIZE}${NC}"
    echo -e "${GREEN}[INFO] NODE_RANK set to: ${NODE_RANK}${NC}"
    echo -e "${GREEN}[INFO] GLOBAL_RANK_OFFSET set to: ${GLOBAL_RANK_OFFSET}${NC}"
else
    ENV_SCRIPT=~/dist_env/$(hostname).sh
    echo -e "${YELLOW}[INFO] Sourcing ${ENV_SCRIPT}.${NC}"
    if [[ -f ${ENV_SCRIPT} ]]; then
        source ${ENV_SCRIPT}
    else
        echo -e "${RED}[ERROR] Environment script ${ENV_SCRIPT} not found!${NC}"

        export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
        export MASTER_PORT=${MASTER_PORT:-32501}
        export WORLD_SIZE=$(( ${SENSECORE_PYTORCH_NNODES:-1} * ${SENSECORE_ACCELERATE_DEVICE_COUNT:-1} ))
        export NODE_RANK=${SENSECORE_PYTORCH_NODE_RANK:-0}
        export GLOBAL_RANK_OFFSET=$(( ${SENSECORE_PYTORCH_NODE_RANK:-0} * ${SENSECORE_ACCELERATE_DEVICE_COUNT:-1} ))

        echo -e "${GREEN}[INFO] MASTER_ADDR set to: ${MASTER_ADDR}${NC}"
        echo -e "${GREEN}[INFO] MASTER_PORT set to: ${MASTER_PORT}${NC}"
        echo -e "${GREEN}[INFO] WORLD_SIZE set to: ${WORLD_SIZE}${NC}"
        echo -e "${GREEN}[INFO] NODE_RANK set to: ${NODE_RANK}${NC}"
        echo -e "${GREEN}[INFO] GLOBAL_RANK_OFFSET set to: ${GLOBAL_RANK_OFFSET}${NC}"
    fi
fi

source ~/proxy.sh

echo -e "${BLUE}Running python $REPO_ROOT/train.py${NC}"
python "$REPO_ROOT/train.py"
