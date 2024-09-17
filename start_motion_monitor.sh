#!/bin/bash

# Description: bash script run command for motion plotting docker container.
# Before running, change INPUT_DIR path string to be the parent directory where input files are located.
# Acceptable input file extensions: *.log (e.g. SLIMM log file), *.tfm or *.txt (e.g. transform files)
# Run command needs to include the specific input filename (with extension) to monitor.
#
# Example run command for SLIMM log file input
# sh start_motion_monitor.sh slimm_2024-08-22_15.06.20.log
#
# Example run command for directory of transform files input
# sh start_motion_monitor.sh navigator_versor001.txt

# Specify local parent directory and grab input file
INPUT_DIR="/path/to/log/files/or/directory/of/transforms/"
INPUT_FILE=$1

# Specify motion calculation variables (i.e. assumed head radius (mm), acceptable motion threshold (mm))
head_radius=50
motion_threshold=0.6

# Build the Docker image, if it does not yet exist
#docker build -t jauger/motion-monitor .

docker run --rm \
  -v $INPUT_DIR:/data \
  jauger/motion-monitor:latest \
  $INPUT_FILE \
  $head_radius \
  $motion_threshold
