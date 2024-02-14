#!/bin/bash

# Define the path to the log file
LOG_FILE="/path/to/log/file.log"

# Get the present working directory
WORKING_DIR=$(pwd)

# Build the Docker image if it doesn't exist
docker build -t log-motion-monitor .

# Run the Docker container with the specified log file path and remove it after completion
docker run --rm -v $WORKING_DIR:/data log-motion-monitor $LOG_FILE
