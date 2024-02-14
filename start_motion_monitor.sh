#!/bin/bash

# Define the path to the log file
LOG_DIR="/home/jauger/Radiology_Research/SLIMM_data/20240209_SLIMM_logs/"

# Get the present working directory
WORKING_DIR=$(pwd)
LOG_FILE=$1

# Build the Docker image, if it doesn't exist
docker build -t log-motion-monitor .

# Run the Docker container with the specified log file path and remove it after completion
docker run --rm \
  -v $LOG_DIR:/data \
  log-motion-monitor:latest \
  $LOG_FILE

