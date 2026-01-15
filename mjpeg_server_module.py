# mjpeg_server.py
import threading
import time
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = "frameboundary"
latest_jpeg = None
lock = threading.Lock()

def generate_dummy_frame(width=1500, height=600, text="Stream connected. Awaiting frames..."):
    """
    Create a white background JPEG frame with centered black text.
    """
    img = 255 * np.ones((height, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
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
        latest_jpeg = generate_dummy_frame()

def update_frame(jpeg_bytes):
    global latest_jpeg
    with lock:
        latest_jpeg = jpeg_bytes

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
                time.sleep(1 / 2)  # client-side pacing only

        except (BrokenPipeError, ConnectionResetError):
            pass

def start_server(port):
    initialize_stream_frame()
    server = ThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
