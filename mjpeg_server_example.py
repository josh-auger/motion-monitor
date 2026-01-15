#!/usr/bin/env python3

# Title: mjpeg_server_example.py

# Description:
# Example test code for streaming images over a MJPEG HTTP streaming server.

# Created on: January 2026
# Created by: Joshua Auger (joshua.auger@childrens.harvard.edu), Computational Radiology Lab, Boston Children's Hospital

import cv2
import time
import numpy as np
import socket
import threading    # used for producer-consumer separation
import glob
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# CONFIG
# ============================================================
IMAGE_DIR = "./test_dataset"
WIDTH = 1500
HEIGHT = 600
FPS = 2
PORT = 8080

BOUNDARY = "frameboundary"

# ============================================================
# SHARED STATE
# ============================================================
latest_jpeg = None
lock = threading.Lock()

# ============================================================
# FRAME PRODUCER
# ============================================================
def generate_dummy_frame(width=1500, height=600, text="Stream connected. Awaiting frames..."):
    """
    Create a white background JPEG frame with centered black text.
    """
    img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(img,text,(text_x, text_y),font,font_scale,(0, 0, 0),thickness,cv2.LINE_AA,)
    success, jpeg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        raise RuntimeError("Failed to generate dummy JPEG frame")
    return jpeg.tobytes()

def initialize_stream_frame():
    global latest_jpeg
    with lock:
        latest_jpeg = generate_dummy_frame(WIDTH, HEIGHT)

def frame_producer():
    global latest_jpeg

    idx = 0

    while True:
        img_list = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

        if not img_list:
            # No frames yet — keep dummy frame alive
            time.sleep(0.5)
            continue

        # Reset index if file list changed or index overflowed
        if idx >= len(img_list):
            idx = 0

        img = cv2.imread(img_list[idx])
        idx += 1

        if img is None:
            time.sleep(0.1)
            continue

        if img.shape[1] != WIDTH or img.shape[0] != HEIGHT:
            img = cv2.resize(img, (WIDTH, HEIGHT))

        success, jpeg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            with lock:
                latest_jpeg = jpeg.tobytes()

        time.sleep(1 / FPS)

# ============================================================
# HTTP MJPEG HANDLER
# ============================================================
class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type",
            f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )
        self.end_headers()

        try:
            while True:
                with lock:
                    frame = latest_jpeg

                if frame is None:
                    time.sleep(0.01)
                    continue

                self.wfile.write(
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                    + frame + b"\r\n"
                )

                time.sleep(1 / FPS)

        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected — clean exit
            pass

# ============================================================
# SERVER BOOTSTRAP
# ============================================================
def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to an external host — no data is sent, just used to determine the local IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def run_server():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), MJPEGHandler)
    host_ip = get_host_ip()
    print(f"MJPEG stream running at: http://{host_ip}:{PORT}/stream.mjpg")
    # If running on crlreconmri SSH server, use crlreconmri IP address: http://10.27.192.112:8080/stream.mjpg
    server.serve_forever()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(IMAGE_DIR, exist_ok=True)

    initialize_stream_frame()   # Initialize stream with dummy frame

    producer_thread = threading.Thread(target=frame_producer, daemon=True)
    producer_thread.start()

    run_server()
