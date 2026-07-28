#!/bin/bash

# Set port number and program variable
PORT=32500
HOST=${MASTER_ADDR:-"localhost"}
HOSTNAME_STR=$(hostname)
LSPROG=$(lsof -i :$PORT)

# Set number of parallel processes per GPU
n=2

source /root/proxy.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pointercad
export no_proxy="localhost,$HOST,127.0.0.1,::1"
export HF_HOME="/mnt/afs_01e/mayi-folder/hf-cache"

# Only check port if no -h argument is provided
if [[ "$HOST" == "$HOSTNAME_STR"* && -z "$LSPROG" ]]; then
  # Port is free, start the test server
  echo "Port $PORT is not in use. Starting test server..."
  python test_server.py -p $PORT
else
  # Port is in use or -h argument is provided, start test-related tasks on GPUs
  echo "Starting tasks on GPUs..."

  NumGPU=$(nvidia-smi --list-gpus | wc -l)

  # Launch tasks on each GPU
  for ((i=0; i<NumGPU; i++)); do
    for ((j=0; j<n; j++)); do
      echo "Launching task $j on GPU $i"
      CUDA_VISIBLE_DEVICES=$i python test.py -h $HOST -p $PORT &
    done
  done

  # Wait for all background tasks to finish
  wait

  echo "All tasks completed."
fi