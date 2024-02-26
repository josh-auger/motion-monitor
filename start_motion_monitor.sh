#!/bin/bash

# Description: bash script run command for motion plotting docker container.
# Before running, change LOG_DIR path string to be the directory where SLIMM log files are saved.
# Run command needs to include the specific log filename to monitor.
#
# Example run command
# sh start_motion_monitor.sh slimm_resting_2024-02-09_17.11.40.log

# Specify local parent directory of log files
LOG_DIR="/home/jauger/Radiology_Research/SLIMM_data/20240209_SLIMM_logs/"

# Get log filename from command line input argument
LOG_FILE=$1

# Build the Docker image, if it does not yet exist
docker build -t jauger/log-motion-monitor .

# Run the Docker container with the specified log file path and remove it after completion
docker run --rm \
  -v $LOG_DIR:/data \
  jauger/log-motion-monitor:latest \
  $LOG_FILE
