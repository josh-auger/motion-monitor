#!/bin/bash

# Define the path to the log file
LOG_FILE="/home/jauger/Radiology_Research/SLIMM_data/20240209_SLIMM_logs/slimm_resting_2024-02-09_17.11.40.log"

# Get the present working directory
WORKING_DIR=$(pwd)

# Build the Docker image, if it doesn't exist
docker build -t log-motion-monitor .

# Run the Docker container with the specified log file path and remove it after completion
docker run --rm -v $WORKING_DIR:/data log-motion-monitor:latest $LOG_FILE
