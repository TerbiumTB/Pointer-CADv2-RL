#!/bin/bash

# Set default port number and host
PORT=32500
HOST="localhost"

# Parse arguments for -p (port) and -h (host)
while getopts "p:h:" opt; do
  case $opt in
    p)
      PORT=$OPTARG
      ;;
    h)
      HOST=$OPTARG
      ;;
  esac
done

# Check if the port is in use
LSPROG=$(lsof -i :$PORT)

# Set number of parallel processes per GPU
n=2

# Only check port if no -h argument is provided
if [[ "$HOST" == "localhost" && -z "$LSPROG" ]]; then
  # Port is free, start the test server
  echo "Port $PORT is not in use. Starting test server..."
  python test_server.py -p $PORT
else
  # Port is in use or host is not localhost, start test-related tasks on GPUs
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
