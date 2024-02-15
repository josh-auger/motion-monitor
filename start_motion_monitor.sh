#!/bin/bash

# Specify local parent directory of log files
LOG_DIR="/home/jauger/Radiology_Research/SLIMM_data/20240209_SLIMM_logs/"

# Get log filename from command line input argument
LOG_FILE=$1

# Build the Docker image, if it doesn't exist
#docker build -t jauger/log-motion-monitor .

# Run the Docker container with the specified log file path and remove it after completion
docker run --rm \
  -v $LOG_DIR:/data \
  jauger/log-motion-monitor:latest \
  $LOG_FILE

