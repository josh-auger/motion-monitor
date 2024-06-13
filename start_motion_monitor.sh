#!/bin/bash

# Description: bash script run command for motion plotting docker container.
# Before running, change INPUT_DIR path string to be the parent directory where input files are located.
# Acceptable input file extensions: *.log (e.g. SLIMM log file), *.tfm or *.txt (e.g. transform files)
# Run command needs to include the specific input filename (with extension) to monitor.
#
# Example run command for SLIMM log file input
# sh start_motion_monitor.sh slimm_2024-04-02_09.21.28.log
#
# Example run command for directory of transform files input
# sh start_motion_monitor.sh navigator_versor001.txt

# Specify local parent directory and grab input file
INPUT_DIR="/home/jauger/Radiology_Research/SLIMM_data/20231207_pre-hemi_4748062_SLIMM_logs/"
#INPUT_DIR="/home_local/ch253208/"  # directory on crlreconmri SSH server
#INPUT_DIR="/home/jauger/Radiology_Research/MPnRAGE_kooshball_data/shot-by-shot_transforms/"  # directory of transform files
INPUT_FILE=$1

# Build the Docker image, if it does not yet exist
#docker build -t jauger/motion-monitor .

docker run --rm \
  -v $INPUT_DIR:/data \
  jauger/motion-monitor:latest \
  $INPUT_FILE
