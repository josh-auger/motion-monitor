#!/usr/bin/env python3

# Title: rtsp_server_example.py

# Description:
# Example test code for streaming images from python OpenCV with ffmpeg

# Created on: January 2026
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

"""
https://stackoverflow.com/questions/68545688/stream-images-from-python-opencv-with-ffmpeg
This example applies the following stages:
- Create 10 synthetic JPEG images in ./test_dataset folder, to be used as input.
- Execute FFplay sub-process as RTSP listener.
    - When using TCP protocol we should start the TCP server first (FFplay is used as a TCP server in out case).
    - We also need the receiver process, because without it, FFmpeg streamer process halts after the first frame.
- Execute FFmpeg sub-process for RTSP streaming.
    - Cyclically read JPEG image to NumPy array (in BGR color format), and write the array as raw video frame to stdin pipe.
Note: It is more efficient to write raw video frames, than encoding each frame to PNG (as used by your reference sample).
"""

import cv2
#import time
import subprocess as sp
import glob
import os


test_path = './test_dataset'  # Folder with synthetic sample images.
os.makedirs(test_path, exist_ok=True)  # Create folder for input images.
os.chdir(test_path)

img_width = 1280
img_height = 720
# Create 10 synthetic JPEG images for testing (image0001.jpg, image0002.jpg, ..., image0010.jpg).
# sp.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', f'testsrc=size={img_width}x{img_height}:rate=1:duration=10', 'image%04d.jpg'])

img_list = sorted(glob.glob("*.jpg"))
img_list_len = len(img_list)
img_index = 0

fps = 1

rtsp_server = "rtsp://0.0.0.0:31415/live.stream"

# Need to start the receiving server first, before the sending client (when using TCP).
# Use FFplay sub-process as an RTSP server to receive images
ffplay_process = sp.Popen(['ffplay', '-rtsp_flags', 'listen', rtsp_server])

# Use FFmpeg sub-process for RTSP streaming client
command = [
    'ffmpeg',
    '-re',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{img_width}x{img_height}',
    '-r', str(fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-rtsp_flags', 'listen',
    '-f', 'rtsp',
    rtsp_server
]
process = sp.Popen(command, stdin=sp.PIPE)

# Main streaming loop
while True:
    current_img = cv2.imread(img_list[img_index])  # Read a JPEG image to NumPy array (in BGR color format) - assume the resolution is correct.
    img_index = (img_index+1) % img_list_len  # Cyclically repeat images

    process.stdin.write(current_img.tobytes())  # Write raw frame to stdin pipe. Sends 1280x720x3 = 2764800 bytes to FFmpeg to stream

    cv2.imshow('current_img', current_img)  # Show image locally for testing
    key = cv2.waitKey(int(round(1000/fps)))  # We need to call cv2.waitKey after cv2.imshow
    if key == 27:  # Press Esc in OpenCV window to exit
        break


process.stdin.close()  # Close stdin pipe
process.wait()  # Wait for FFmpeg sub-process to finish
ffplay_process.kill()  # Forcefully close FFplay sub-process
cv2.destroyAllWindows()  # Close OpenCV window