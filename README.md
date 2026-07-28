# Pointer-CAD Source Code

This repository contains the core code and configuration files for the Pointer-CAD project. It supports CAD model representation, data processing, training, and evaluation.

## Directory Structure

conda create --name pointercad-rl --clone pointercad


* `cadmodel`
  Contains class definitions for CAD models and implementations of core geometric operations, such as sketches, extrusions, fillets, and chamfers. It is used to represent and transform CAD geometry.

* `config`
  Contains configuration files used for training and experiments, including hyperparameters, model settings, paths, and other options. These files make it easier to reproduce experiments and switch between different strategies.

* `dataset`
  Contains dataset-related code, including data loaders, preprocessing, and data augmentations. It is responsible for building the training, validation, and test sets and providing batch-processing interfaces.

* `measurement`
  Contains evaluation metrics and scripts used during evaluation, including quantitative methods for error calculation and similarity assessment.

* `metrics`
  Contains reinforcement learning–related reward functions and metric implementations. It encapsulates different reward designs and statistical utilities used by training strategies.

* `models`
  Contains PyTorch model implementations, including network architectures, forward inference, and model saving and loading. These modules are used together with the training scripts.

---

## Usage Instructions Based on `train.sh`

### Prerequisites and Setup

* A Bash environment and a suitable Python installation are required.

* When using GPUs, `nvidia-smi` must be available because test mode uses it to automatically detect the number of GPUs.

* In non-test mode, a per-node environment script must be prepared at:

  ```bash
  ~/dist_env/$(hostname).sh
  ```

  The training script sources this file to configure distributed environment variables.

* `train.sh` attempts to source the following proxy script:

  ```bash
  ~/cluster/set_proxy.sh
  ```

  Remove or override this path if a proxy is not required.

* Make the script executable:

  ```bash
  chmod +x train.sh
  ```

### Script Behavior Overview

#### Test Mode for Quick Debugging

Run:

```bash
./train.sh --test
```

or:

```bash
./train.sh -t
```

The script sets the following variables:

```bash
MASTER_ADDR=localhost
MASTER_PORT=32501
WORLD_SIZE=$(nvidia-smi --list-gpus | wc -l)
NODE_RANK=0
GLOBAL_RANK_OFFSET=0
```

This mode is suitable for single-machine debugging or quick CI validation.

#### Normal Mode for Cluster or Multi-Node Training

The script attempts to source:

```bash
~/dist_env/$(hostname).sh
```

If the file does not exist, the script exits with an error.

### Usage Examples

Local quick debugging:

```bash
./train.sh --test
```

Running in a prepared cluster environment:

```bash
./train.sh
```

At the end, the script runs:

```bash
source ~/cluster/set_proxy.sh  # if available
python ./train.py
```

---

## Helper Scripts

### `set_proxy.sh`

```bash
#!/usr/bin/env bash
DEFAULT_PROXY="http://10.10.10.102:7999"
export http_proxy="$DEFAULT_PROXY"
export https_proxy="$DEFAULT_PROXY"
echo "Proxy has been set:"
echo "http_proxy=$http_proxy"
echo "https_proxy=$https_proxy"
```

### Slurm Job Script

The following example is intended for a distributed training environment and generates a per-node environment script for each allocated node.

```bash
#!/bin/bash
#SBATCH -J cad_gpu_multi
#SBATCH -o log/cad_gpu_multi_%j.out
#SBATCH -e log/cad_gpu_multi_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=192
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00

# Get the allocated nodes
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

# Set the primary node address and port
MASTER_ADDR=${nodes_array[0]}
MASTER_PORT=32501

# Get the number of GPUs on each node
GPUS_PER_NODE=$SLURM_GPUS_PER_NODE

# Calculate the total world size
WORLD_SIZE=$((${#nodes_array[@]} * $GPUS_PER_NODE))

# Create the directory for environment scripts
env_dir=~/dist_env
mkdir -p "$env_dir"

# Generate an environment script for each node
for i in "${!nodes_array[@]}"; do
  node=${nodes_array[$i]}
  rank_offset=$(($i * $GPUS_PER_NODE))
  node_script="$env_dir/${node}.sh"

  cat <<EOF > "$node_script"
#!/bin/bash
# Auto-generated environment configuration for $node

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$WORLD_SIZE
export GPUS_PER_NODE=$GPUS_PER_NODE
export NODE_RANK=$i
export GLOBAL_RANK_OFFSET=$rank_offset
EOF

  chmod +x "$node_script"
done

echo "Generated per-node environment scripts in $env_dir"

# Keep the job allocation active
tail -f /dev/null
```
