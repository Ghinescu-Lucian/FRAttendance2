import os
import sys
import time

# Must be configured before importing cv2 / before opening the stream.
os.environ.setdefault(
    'OPENCV_FFMPEG_CAPTURE_OPTIONS',
    'rtsp_transport;tcp|rtsp_flags;prefer_tcp|stimeout;3000000|rw_timeout;3000000|timeout;3000000|max_delay;200000|reorder_queue_size;0|fflags;nobuffer+discardcorrupt|flags;low_delay|probesize;32768|analyzeduration;0',
)
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')

import cv2

if len(sys.argv) < 2:
    print('Usage: py rtsp_fps_test.py "rtsp://user:password@ip:554/Streaming/channels/102"')
    raise SystemExit(2)

source = sys.argv[1]
print('Opening:', source)
print('OPENCV_FFMPEG_CAPTURE_OPTIONS=', os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS'))

cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print('ERROR: could not open stream')
    raise SystemExit(1)

try:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
except Exception:
    pass

frames = 0
failures = 0
first_shape = None
start = time.time()
last_report = start
while time.time() - start < 15.0:
    ok, frame = cap.read()
    if not ok or frame is None:
        failures += 1
        time.sleep(0.01)
        continue
    frames += 1
    if first_shape is None:
        h, w = frame.shape[:2]
        first_shape = f'{w}x{h}'
    now = time.time()
    if now - last_report >= 5.0:
        elapsed = now - start
        print(f'partial: {frames / elapsed:.2f} FPS | frames={frames} | failures={failures} | size={first_shape}')
        last_report = now

elapsed = max(0.001, time.time() - start)
print(f'RAW_RTSP_FPS={frames / elapsed:.2f}')
print(f'frames={frames}')
print(f'failures={failures}')
print(f'size={first_shape}')
cap.release()
