import os
import re
import time
import csv
import json
import urllib.request
import urllib.error
import ssl
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

cv2.setUseOptimized(True)

MANUAL_ZOOM_CONTROL_PANEL_VERSION = "2026-05-23-visible-sliders-v1"
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass


def load_environment_file(env_path=".env"):
    """
    Load settings from a .env file before configuration constants read os.getenv().
    Uses python-dotenv if installed, otherwise falls back to a tiny local parser.
    Existing OS environment variables win over .env values.
    """
    path = Path(env_path)

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(path)
        return
    except Exception:
        pass

    if not path.exists():
        return

    try:
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"[WARN] Could not load .env file: {exc}")


load_environment_file()


# ============================================================
# FAST CLEAN MANY-FACE DETECTION: OpenCV YuNet + SFace
# ============================================================
# Goal:
#   1. Detect as many faces as possible in the camera frame.
#   2. Recognize every detected face independently.
#   3. Draw every visible face on the full camera view.
#   4. Mark attendance per person after that person is stable.
#   5. Use full-frame detection most of the time for speed.
#   6. Use multi-scale/grid crop search only occasionally to help small or far faces.
#   7. Suppress duplicate Unknown boxes around an already-known face.
#
# This version uses:
#   YuNet = cv2.FaceDetectorYN
#   SFace = cv2.FaceRecognizerSF
#
# Install:
#   py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
#   py -m pip install opencv-contrib-python numpy
#
# Run:
#   py main_yunet_sface_many_faces_moodle_integrated.py
# ============================================================


# =========================
# Configuration
# =========================
CAMERA_INDEX = 0
# Folder or file containing precomputed embeddings.
# You can point this to:
#   - one file:   "images/GhinescuLucian_embeddings_1.json"
#   - a folder:   "images/"
# The loader accepts files named like *_embeddings*, embeddings_*.json, or normal .json files.
EMBEDDINGS_SOURCE = "images/"

# Kept only if you still want the old image-loading function for debugging.
ENCODINGS_DIR = "images/"
CAPTURE_DIR = "captures"
WINDOW_NAME = "YuNet + SFace Fast Many Faces"

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

UNKNOWN_LABEL = "Unknown"

# Models
MODELS_DIR = Path("models")
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# YuNet/SFace thresholds
YUNET_SCORE_THRESHOLD = 0.70
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 1000

# SFace cosine similarity threshold.
# Lower = accepts more matches but more risk of wrong label.
# Higher = stricter but more Unknown.
SFACE_SIMILARITY_THRESHOLD = 0.36

# Display zoom after a face target is found.
MIN_ZOOM = 1.0
MAX_ZOOM = 3.5
FACE_TARGET_HEIGHT = 0.38

CENTER_SMOOTHING = 0.22
ZOOM_SMOOTHING = 0.16

# Recognition input size for zoom-search crops.
# Lower = faster, but weaker far-face detection.
SEARCH_INPUT_WIDTH = 640

# Search behavior.
FAST_SEARCH_EVERY_N_FRAMES = 1
FULL_GRID_SEARCH_EVERY_N_FRAMES = 60
ENABLE_PERIODIC_GRID_SEARCH = False

# Search zooms.
SEARCH_ZOOM_LEVELS_FAST = [1.0]
SEARCH_ZOOM_LEVELS_FULL = [1.0, 1.8]

# Tracking/follow behavior.
STABLE_FRAMES_REQUIRED = 5
PHOTO_COOLDOWN_SECONDS = 8
MAX_LOST_FRAMES_AFTER_LOCK = 12

# Multi-face behavior.
# The program still chooses one face only for the display zoom/focus,
# but recognition, drawing, stability, photos, and attendance are handled
# independently for every visible known face.
FACE_STATE_TIMEOUT_SECONDS = 2.0
CANDIDATE_IOU_DEDUPE_THRESHOLD = 0.30
DRAW_ALL_RECOGNIZED_FACES = True
# False by default because duplicate crop detections can create confusing
# red Unknown labels over a correctly recognized known person.
DRAW_UNKNOWN_FACES = True
SUPPRESS_UNKNOWN_NEAR_KNOWN = True
UNKNOWN_SUPPRESSION_IOU = 0.10
UNKNOWN_SUPPRESSION_CENTER_RATIO = 0.75
SAME_FACE_CENTER_RATIO = 0.55

# Fast UI labels. Keep labels short so the image is readable with many faces.
SHORT_LABELS = True
SHORT_LABEL_MAX_CHARS = 10
UNKNOWN_SHORT_LABEL = "UNK"
DRAW_SIMILARITY_ON_LABEL = False
DRAW_STABILITY_ON_LABEL = True

# Save policy.
SAVE_PHOTO_WHEN_KNOWN_STABLE = True
MAX_PHOTOS_PER_PERSON = 1

DRAW_DEBUG_SEARCH_SOURCE = False

# Entrance/door mode.
# The program will only choose a face whose CENTER is inside this zone.
# Values are percentages of the camera frame: 0.0 = left/top, 1.0 = right/bottom.
ENTRANCE_MODE = False
ENTRANCE_ZONE_X1 = 0.25
ENTRANCE_ZONE_Y1 = 0.08
ENTRANCE_ZONE_X2 = 0.75
ENTRANCE_ZONE_Y2 = 0.95
DRAW_ENTRANCE_ZONE = False

# In a doorway queue, the closest/front person usually has the biggest face box.
ENTRANCE_SORT_BY_BIGGEST_FACE = True

# Attendance settings.
ATTENDANCE_CSV = "attendance.csv"
ATTENDANCE_COOLDOWN_SECONDS = 60
STOP_FOLLOWING_AFTER_ATTENDANCE_SECONDS = 2.0

# Moodle FaceAttendance API settings.
# These defaults match the integration details from the Moodle plugin page.
# You can also override them from environment variables without editing code:
#   set FACE_ATTENDANCE_SECRET=your-real-secret
#   set FACE_ATTENDANCE_MARK_URL=https://your-host/mod/faceattendance/api/mark.php
#   set FACE_ATTENDANCE_ROSTER_URL=https://your-host/mod/faceattendance/api/roster.php?cmid=18
FACE_ATTENDANCE_MARK_ENABLED = os.getenv("FACE_ATTENDANCE_MARK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
FACE_ATTENDANCE_MARK_URL = os.getenv(
    "FACE_ATTENDANCE_MARK_URL",
    "https://192.168.0.154/mod/faceattendance/api/mark.php",
)
FACE_ATTENDANCE_ROSTER_URL = os.getenv(
    "FACE_ATTENDANCE_ROSTER_URL",
    "https://192.168.0.154/mod/faceattendance/api/roster.php?cmid=18",
)
FACE_ATTENDANCE_CMID = int(os.getenv("FACE_ATTENDANCE_CMID", "18"))
FACE_ATTENDANCE_SECRET = os.getenv("FACE_ATTENDANCE_SECRET", "your-api-secret-here")
FACE_ATTENDANCE_SOURCE = os.getenv("FACE_ATTENDANCE_SOURCE", "camera-lab-1")
FACE_ATTENDANCE_TIMEOUT_SECONDS = float(os.getenv("FACE_ATTENDANCE_TIMEOUT_SECONDS", "3.0"))

# Local HTTPS test servers often use self-signed certificates or an IP-address URL.
# For real production HTTPS, set this to true and install a valid certificate.
FACE_ATTENDANCE_VERIFY_TLS = os.getenv("FACE_ATTENDANCE_VERIFY_TLS", "false").strip().lower() in {"1", "true", "yes", "on"}

# Optional mapping from your local recognition names to Moodle externalid values.
# If your embedding file/person name is already the Moodle externalid, leave this empty.
# Example:
# FACE_EXTERNAL_ID_BY_NAME = {
#     "GhinescuLucian": "face_student_001",
# }
FACE_EXTERNAL_ID_BY_NAME = {
}

# Manual digital zoom for aiming the processing view at the classroom entrance.
# This is a software crop/resize zoom, so it works with normal webcams too.
# It does not change the physical/optical zoom of the camera.
MANUAL_ENTRANCE_ZOOM_ENABLED = os.getenv("MANUAL_ENTRANCE_ZOOM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
MANUAL_ENTRANCE_INITIAL_ZOOM = float(os.getenv("MANUAL_ENTRANCE_INITIAL_ZOOM", "1.0"))
MANUAL_ENTRANCE_MAX_ZOOM = float(os.getenv("MANUAL_ENTRANCE_MAX_ZOOM", "6.0"))
MANUAL_ENTRANCE_CENTER_X = float(os.getenv("MANUAL_ENTRANCE_CENTER_X", "0.5"))
MANUAL_ENTRANCE_CENTER_Y = float(os.getenv("MANUAL_ENTRANCE_CENTER_Y", "0.5"))
MANUAL_ENTRANCE_ZOOM_STEP = float(os.getenv("MANUAL_ENTRANCE_ZOOM_STEP", "0.15"))
MANUAL_ENTRANCE_PAN_STEP_RATIO = float(os.getenv("MANUAL_ENTRANCE_PAN_STEP_RATIO", "0.08"))
DRAW_MANUAL_ZOOM_HELP = os.getenv("DRAW_MANUAL_ZOOM_HELP", "true").strip().lower() in {"1", "true", "yes", "on"}
DRAW_MANUAL_ZOOM_CROSSHAIR = os.getenv("DRAW_MANUAL_ZOOM_CROSSHAIR", "true").strip().lower() in {"1", "true", "yes", "on"}

# Visible OpenCV slider panel for manual entrance zoom.
# This is the part you asked for: not only hidden keyboard shortcuts.
# It creates a separate window named "Manual Zoom Controls" with sliders.
MANUAL_ENTRANCE_SHOW_CONTROL_PANEL = os.getenv("MANUAL_ENTRANCE_SHOW_CONTROL_PANEL", "true").strip().lower() in {"1", "true", "yes", "on"}
MANUAL_ENTRANCE_CONTROL_PANEL_NAME = os.getenv("MANUAL_ENTRANCE_CONTROL_PANEL_NAME", "Manual Zoom Controls")


# Automatic door/entrance zoom.
# This does not use a trained semantic "door detector". It watches the raw camera
# image for a stable large motion region, which is exactly what usually happens
# when a classroom door opens or a person enters through it. When found, it moves
# the manual digital zoom crop to that region.
AUTO_DOOR_ZOOM_ENABLED = os.getenv("AUTO_DOOR_ZOOM_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
AUTO_DOOR_ZOOM_TARGET_ZOOM = float(os.getenv("AUTO_DOOR_ZOOM_TARGET_ZOOM", "2.4"))
AUTO_DOOR_ZOOM_MIN_AREA_RATIO = float(os.getenv("AUTO_DOOR_ZOOM_MIN_AREA_RATIO", "0.018"))
AUTO_DOOR_ZOOM_MAX_AREA_RATIO = float(os.getenv("AUTO_DOOR_ZOOM_MAX_AREA_RATIO", "0.70"))
AUTO_DOOR_ZOOM_STABLE_FRAMES = int(os.getenv("AUTO_DOOR_ZOOM_STABLE_FRAMES", "5"))
AUTO_DOOR_ZOOM_UPDATE_COOLDOWN_SECONDS = float(os.getenv("AUTO_DOOR_ZOOM_UPDATE_COOLDOWN_SECONDS", "2.0"))
AUTO_DOOR_ZOOM_LOCK_AFTER_FOUND = os.getenv("AUTO_DOOR_ZOOM_LOCK_AFTER_FOUND", "true").strip().lower() in {"1", "true", "yes", "on"}
AUTO_DOOR_ZOOM_RESIZE_WIDTH = int(os.getenv("AUTO_DOOR_ZOOM_RESIZE_WIDTH", "480"))
AUTO_DOOR_ZOOM_MOTION_THRESHOLD = int(os.getenv("AUTO_DOOR_ZOOM_MOTION_THRESHOLD", "200"))
DRAW_AUTO_DOOR_ZOOM_DEBUG = os.getenv("DRAW_AUTO_DOOR_ZOOM_DEBUG", "true").strip().lower() in {"1", "true", "yes", "on"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


os.makedirs(CAPTURE_DIR, exist_ok=True)


# ============================================================
# Model setup
# ============================================================

def ensure_model_file(path: Path, url: str):
    MODELS_DIR.mkdir(exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        return

    print(f"Downloading model: {path.name}")
    print(url)
    urllib.request.urlretrieve(url, str(path))

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to download model: {path}")


def check_opencv_api():
    missing = []

    if not hasattr(cv2, "FaceDetectorYN_create"):
        missing.append("cv2.FaceDetectorYN_create")

    if not hasattr(cv2, "FaceRecognizerSF_create"):
        missing.append("cv2.FaceRecognizerSF_create")

    if missing:
        raise RuntimeError(
            "Your OpenCV build does not include YuNet/SFace APIs:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nInstall OpenCV contrib:\n"
            "  py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n"
            "  py -m pip install opencv-contrib-python numpy\n"
        )


def create_detector(input_size: Tuple[int, int] = (320, 320)):
    return cv2.FaceDetectorYN_create(
        str(YUNET_MODEL),
        "",
        input_size,
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K,
    )


def create_recognizer():
    return cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")


# ============================================================
# Zoom controller: same principle as first program
# ============================================================

class FaceCenterZoomer:
    def __init__(self):
        self.center_x = None
        self.center_y = None
        self.zoom = MIN_ZOOM

    def _clamp_center(self, cx, cy, frame_w, frame_h, zoom):
        crop_w = frame_w / zoom
        crop_h = frame_h / zoom
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        cx = max(half_w, min(frame_w - half_w, cx))
        cy = max(half_h, min(frame_h - half_h, cy))

        return cx, cy

    def update(self, frame_shape, face_box=None):
        frame_h, frame_w = frame_shape[:2]

        if self.center_x is None or self.center_y is None:
            self.center_x = frame_w / 2.0
            self.center_y = frame_h / 2.0

        if face_box is None:
            # Do not jump back to full frame immediately.
            target_cx = self.center_x
            target_cy = self.center_y
            target_zoom = max(MIN_ZOOM, self.zoom * 0.985)
        else:
            x1, y1, x2, y2 = face_box
            face_h = max(1, y2 - y1)

            target_cx = (x1 + x2) / 2.0
            target_cy = (y1 + y2) / 2.0
            target_zoom = (FACE_TARGET_HEIGHT * frame_h) / face_h
            target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, target_zoom))

        self.center_x = (1.0 - CENTER_SMOOTHING) * self.center_x + CENTER_SMOOTHING * target_cx
        self.center_y = (1.0 - CENTER_SMOOTHING) * self.center_y + CENTER_SMOOTHING * target_cy
        self.zoom = (1.0 - ZOOM_SMOOTHING) * self.zoom + ZOOM_SMOOTHING * target_zoom

        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))

        self.center_x, self.center_y = self._clamp_center(
            self.center_x,
            self.center_y,
            frame_w,
            frame_h,
            self.zoom,
        )

        return self.get_crop_params(frame_w, frame_h)

    def get_crop_params(self, frame_w, frame_h):
        crop_w = int(frame_w / self.zoom)
        crop_h = int(frame_h / self.zoom)

        left = int(round(self.center_x - crop_w / 2.0))
        top = int(round(self.center_y - crop_h / 2.0))

        left = max(0, min(frame_w - crop_w, left))
        top = max(0, min(frame_h - crop_h, top))

        return left, top, crop_w, crop_h

    @staticmethod
    def apply_zoom(frame, crop_params):
        frame_h, frame_w = frame.shape[:2]
        left, top, crop_w, crop_h = crop_params
        crop = frame[top:top + crop_h, left:left + crop_w]
        return cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


class ManualEntranceZoomController:
    """
    Manual software pan/zoom controller for a fixed entrance/door camera.

    The program crops the original webcam frame around this controller's center,
    resizes that crop back to the normal camera resolution, and then runs face
    detection/recognition on the zoomed view. This makes the entrance area larger
    for YuNet/SFace while ignoring irrelevant parts of the room.
    """

    def __init__(self, frame_w, frame_h):
        self.enabled = bool(MANUAL_ENTRANCE_ZOOM_ENABLED)
        self.zoom = self._clamp_zoom(MANUAL_ENTRANCE_INITIAL_ZOOM)
        self.center_x_ratio = self._clamp_ratio(MANUAL_ENTRANCE_CENTER_X)
        self.center_y_ratio = self._clamp_ratio(MANUAL_ENTRANCE_CENTER_Y)
        self._last_crop_params = (0, 0, max(1, int(frame_w)), max(1, int(frame_h)))
        self._last_print_time = 0.0

    @staticmethod
    def _clamp_ratio(value):
        try:
            value = float(value)
        except Exception:
            value = 0.5
        return max(0.0, min(1.0, value))

    @staticmethod
    def _clamp_zoom(value):
        try:
            value = float(value)
        except Exception:
            value = 1.0
        return max(1.0, min(float(MANUAL_ENTRANCE_MAX_ZOOM), value))

    def get_crop_params(self, frame_w, frame_h):
        frame_w = max(1, int(frame_w))
        frame_h = max(1, int(frame_h))

        if not self.enabled or self.zoom <= 1.0001:
            self._last_crop_params = (0, 0, frame_w, frame_h)
            return self._last_crop_params

        crop_w = max(1, int(round(frame_w / self.zoom)))
        crop_h = max(1, int(round(frame_h / self.zoom)))

        cx = self.center_x_ratio * frame_w
        cy = self.center_y_ratio * frame_h

        left = int(round(cx - crop_w / 2.0))
        top = int(round(cy - crop_h / 2.0))

        left = max(0, min(frame_w - crop_w, left))
        top = max(0, min(frame_h - crop_h, top))

        # If the crop was clamped at an edge, keep the stored center honest.
        actual_cx = left + crop_w / 2.0
        actual_cy = top + crop_h / 2.0
        self.center_x_ratio = self._clamp_ratio(actual_cx / float(frame_w))
        self.center_y_ratio = self._clamp_ratio(actual_cy / float(frame_h))

        self._last_crop_params = (left, top, crop_w, crop_h)
        return self._last_crop_params

    def apply_to_frame(self, frame):
        frame_h, frame_w = frame.shape[:2]
        crop_params = self.get_crop_params(frame_w, frame_h)
        left, top, crop_w, crop_h = crop_params

        if not self.enabled or self.zoom <= 1.0001:
            return frame.copy(), crop_params

        crop = frame[top:top + crop_h, left:left + crop_w]
        zoomed_frame = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        return zoomed_frame, crop_params

    def reset(self):
        self.zoom = 1.0
        self.center_x_ratio = 0.5
        self.center_y_ratio = 0.5
        print("Manual entrance zoom reset to full frame.")

    def set_target(self, center_x_ratio, center_y_ratio, zoom=None, enabled=True):
        """Programmatically aim the processing crop at a normalized camera position."""
        self.enabled = bool(enabled)
        self.center_x_ratio = self._clamp_ratio(center_x_ratio)
        self.center_y_ratio = self._clamp_ratio(center_y_ratio)
        if zoom is not None:
            self.zoom = self._clamp_zoom(zoom)
        return True

    def zoom_by(self, delta):
        old_zoom = self.zoom
        self.zoom = self._clamp_zoom(self.zoom + delta)
        return abs(self.zoom - old_zoom) > 1e-6

    def pan_by(self, dx, dy, frame_w, frame_h):
        old_x = self.center_x_ratio
        old_y = self.center_y_ratio

        crop_params = self.get_crop_params(frame_w, frame_h)
        _, _, crop_w, crop_h = crop_params

        # Step is proportional to the visible crop. At high zoom, pan in smaller movements.
        self.center_x_ratio = self._clamp_ratio(
            self.center_x_ratio + dx * MANUAL_ENTRANCE_PAN_STEP_RATIO * (crop_w / float(max(1, frame_w)))
        )
        self.center_y_ratio = self._clamp_ratio(
            self.center_y_ratio + dy * MANUAL_ENTRANCE_PAN_STEP_RATIO * (crop_h / float(max(1, frame_h)))
        )
        self.get_crop_params(frame_w, frame_h)

        return abs(self.center_x_ratio - old_x) > 1e-6 or abs(self.center_y_ratio - old_y) > 1e-6

    def toggle(self):
        self.enabled = not self.enabled
        print(f"Manual entrance zoom: {'ON' if self.enabled else 'OFF'}")
        return True

    def print_current_env_values(self):
        print("\nCurrent manual entrance zoom values for .env:")
        print(f"MANUAL_ENTRANCE_ZOOM_ENABLED={'true' if self.enabled else 'false'}")
        print(f"MANUAL_ENTRANCE_INITIAL_ZOOM={self.zoom:.3f}")
        print(f"MANUAL_ENTRANCE_CENTER_X={self.center_x_ratio:.4f}")
        print(f"MANUAL_ENTRANCE_CENTER_Y={self.center_y_ratio:.4f}")
        print(f"MANUAL_ENTRANCE_MAX_ZOOM={MANUAL_ENTRANCE_MAX_ZOOM}")
        print()

    def handle_key(self, key_raw, frame_w, frame_h):
        """Return True when the visible processing crop changed."""
        key = key_raw & 0xFF

        # Windows waitKeyEx arrow codes plus common Linux/X11 arrow codes.
        left_keys = {2424832, 65361}
        up_keys = {2490368, 65362}
        right_keys = {2555904, 65363}
        down_keys = {2621440, 65364}

        changed = False

        if key in (ord("m"), ord("M")):
            return self.toggle()

        if key in (ord("z"), ord("Z")):
            self.reset()
            return True

        if key in (ord("p"), ord("P")):
            self.print_current_env_values()
            return False

        if key in (ord("+"), ord("=")):
            changed = self.zoom_by(MANUAL_ENTRANCE_ZOOM_STEP)
        elif key in (ord("-"), ord("_")):
            changed = self.zoom_by(-MANUAL_ENTRANCE_ZOOM_STEP)
        elif key in (ord("a"), ord("A")) or key_raw in left_keys:
            changed = self.pan_by(-1, 0, frame_w, frame_h)
        elif key in (ord("d"), ord("D")) or key_raw in right_keys:
            changed = self.pan_by(1, 0, frame_w, frame_h)
        elif key in (ord("w"), ord("W")) or key_raw in up_keys:
            changed = self.pan_by(0, -1, frame_w, frame_h)
        elif key in (ord("s"), ord("S")) or key_raw in down_keys:
            changed = self.pan_by(0, 1, frame_w, frame_h)

        if changed:
            now = time.time()
            if now - self._last_print_time > 0.35:
                print(
                    "Manual zoom crop: "
                    f"zoom={self.zoom:.2f}x center=({self.center_x_ratio:.3f}, {self.center_y_ratio:.3f})"
                )
                self._last_print_time = now

        return changed

    @staticmethod
    def _noop_trackbar_callback(_value):
        # OpenCV requires a callback for trackbars, but we read slider values in the main loop.
        pass

    def create_control_panel(self):
        """
        Create a visible slider window for manual entrance zoom.

        Sliders:
          - Manual ON: 0/1
          - Zoom x100: 100 = 1.00x, 240 = 2.40x, etc.
          - Center X %: 0..100
          - Center Y %: 0..100
        """
        if not MANUAL_ENTRANCE_SHOW_CONTROL_PANEL:
            return

        self._control_panel_ready = False
        self._syncing_control_panel = False
        self._last_control_panel_state = None

        cv2.namedWindow(MANUAL_ENTRANCE_CONTROL_PANEL_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(MANUAL_ENTRANCE_CONTROL_PANEL_NAME, 520, 180)

        max_zoom_x100 = max(100, int(round(float(MANUAL_ENTRANCE_MAX_ZOOM) * 100)))
        initial_zoom_x100 = max(100, min(max_zoom_x100, int(round(self.zoom * 100))))
        initial_x = max(0, min(100, int(round(self.center_x_ratio * 100))))
        initial_y = max(0, min(100, int(round(self.center_y_ratio * 100))))

        cv2.createTrackbar("Manual ON", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, 1 if self.enabled else 0, 1, self._noop_trackbar_callback)
        cv2.createTrackbar("Zoom x100", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, initial_zoom_x100, max_zoom_x100, self._noop_trackbar_callback)
        cv2.createTrackbar("Center X %", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, initial_x, 100, self._noop_trackbar_callback)
        cv2.createTrackbar("Center Y %", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, initial_y, 100, self._noop_trackbar_callback)

        self._control_panel_ready = True
        self.sync_control_panel_to_state()
        print('Visible manual zoom controls opened in the "Manual Zoom Controls" window.')
        print('Use the sliders there, or use keyboard: +/- zoom, WASD/arrows pan, M toggle, Z reset, P print values.')

    def _read_control_panel_state(self):
        if not getattr(self, "_control_panel_ready", False):
            return None

        try:
            enabled_pos = cv2.getTrackbarPos("Manual ON", MANUAL_ENTRANCE_CONTROL_PANEL_NAME)
            zoom_x100 = cv2.getTrackbarPos("Zoom x100", MANUAL_ENTRANCE_CONTROL_PANEL_NAME)
            center_x_percent = cv2.getTrackbarPos("Center X %", MANUAL_ENTRANCE_CONTROL_PANEL_NAME)
            center_y_percent = cv2.getTrackbarPos("Center Y %", MANUAL_ENTRANCE_CONTROL_PANEL_NAME)
        except cv2.error:
            return None

        max_zoom_x100 = max(100, int(round(float(MANUAL_ENTRANCE_MAX_ZOOM) * 100)))
        zoom_x100 = max(100, min(max_zoom_x100, int(zoom_x100)))

        return (
            1 if enabled_pos else 0,
            zoom_x100,
            max(0, min(100, int(center_x_percent))),
            max(0, min(100, int(center_y_percent))),
        )

    def sync_control_panel_to_state(self):
        """Push current internal zoom state to the visible sliders."""
        if not getattr(self, "_control_panel_ready", False):
            return

        try:
            self._syncing_control_panel = True
            max_zoom_x100 = max(100, int(round(float(MANUAL_ENTRANCE_MAX_ZOOM) * 100)))
            cv2.setTrackbarPos("Manual ON", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, 1 if self.enabled else 0)
            cv2.setTrackbarPos("Zoom x100", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, max(100, min(max_zoom_x100, int(round(self.zoom * 100)))))
            cv2.setTrackbarPos("Center X %", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, max(0, min(100, int(round(self.center_x_ratio * 100)))))
            cv2.setTrackbarPos("Center Y %", MANUAL_ENTRANCE_CONTROL_PANEL_NAME, max(0, min(100, int(round(self.center_y_ratio * 100)))))
            self._last_control_panel_state = self._read_control_panel_state()
        except cv2.error:
            pass
        finally:
            self._syncing_control_panel = False

    def update_from_control_panel(self, frame_w, frame_h):
        """
        Pull slider changes into the zoom controller.
        Returns True when the crop changed because the user moved a slider.
        """
        if not getattr(self, "_control_panel_ready", False):
            return False
        if getattr(self, "_syncing_control_panel", False):
            return False

        state = self._read_control_panel_state()
        if state is None:
            return False

        if self._last_control_panel_state is None:
            self._last_control_panel_state = state
            return False

        if state == self._last_control_panel_state:
            return False

        enabled_pos, zoom_x100, center_x_percent, center_y_percent = state

        old = (self.enabled, self.zoom, self.center_x_ratio, self.center_y_ratio)

        self.enabled = bool(enabled_pos)
        self.zoom = self._clamp_zoom(zoom_x100 / 100.0)
        self.center_x_ratio = self._clamp_ratio(center_x_percent / 100.0)
        self.center_y_ratio = self._clamp_ratio(center_y_percent / 100.0)
        self.get_crop_params(frame_w, frame_h)

        self._last_control_panel_state = self._read_control_panel_state()

        changed = (
            old[0] != self.enabled
            or abs(old[1] - self.zoom) > 1e-6
            or abs(old[2] - self.center_x_ratio) > 1e-6
            or abs(old[3] - self.center_y_ratio) > 1e-6
        )

        if changed:
            print(
                "Manual zoom slider changed: "
                f"enabled={'ON' if self.enabled else 'OFF'}, "
                f"zoom={self.zoom:.2f}x, "
                f"center=({self.center_x_ratio:.3f}, {self.center_y_ratio:.3f})"
            )

        return changed

    def draw_overlay(self, display_frame, crop_params):
        frame_h, frame_w = display_frame.shape[:2]

        if DRAW_MANUAL_ZOOM_CROSSHAIR and self.enabled:
            cx = frame_w // 2
            cy = frame_h // 2
            cv2.line(display_frame, (cx - 18, cy), (cx + 18, cy), (255, 255, 255), 1)
            cv2.line(display_frame, (cx, cy - 18), (cx, cy + 18), (255, 255, 255), 1)

        if not DRAW_MANUAL_ZOOM_HELP:
            return

        left, top, crop_w, crop_h = crop_params
        mode = "ON" if self.enabled else "OFF"
        text = (
            f"Manual entrance zoom {mode} | {self.zoom:.2f}x | "
            f"center {self.center_x_ratio:.2f},{self.center_y_ratio:.2f} | "
            "SLIDERS WINDOW + keyboard: +/- zoom | WASD/arrows pan | M toggle | Z reset | P print .env"
        )

        cv2.putText(
            display_frame,
            text,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        if self.enabled and self.zoom > 1.0001:
            crop_text = f"Source crop: x={left} y={top} w={crop_w} h={crop_h}"
            cv2.putText(
                display_frame,
                crop_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )


class AutoDoorZoomController:
    """
    Auto-aims the manual entrance zoom based on stable motion in the raw frame.

    This is intentionally simple and fast:
      - Build a foreground/motion mask with MOG2.
      - Keep large motion boxes only.
      - Require the motion center to remain stable for a few frames.
      - Move the digital zoom crop to that target.

    It is useful for a classroom doorway because the door opening and students
    entering create the strongest stable motion region in a mostly static camera.
    """

    def __init__(self):
        self.enabled = bool(AUTO_DOOR_ZOOM_ENABLED)
        self.locked = False
        self.last_motion_box = None
        self.last_target = None
        self.stable_count = 0
        self.last_apply_time = 0.0
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120,
            varThreshold=32,
            detectShadows=True,
        )

    @staticmethod
    def _box_center_ratio(box, frame_w, frame_h):
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / float(max(1, frame_w))
        cy = ((y1 + y2) / 2.0) / float(max(1, frame_h))
        return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))

    @staticmethod
    def _center_distance_ratio(box_a, box_b, frame_w, frame_h):
        ax, ay = AutoDoorZoomController._box_center_ratio(box_a, frame_w, frame_h)
        bx, by = AutoDoorZoomController._box_center_ratio(box_b, frame_w, frame_h)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def toggle(self):
        self.enabled = not self.enabled
        print(f"Auto door zoom: {'ON' if self.enabled else 'OFF'}")

    def toggle_lock(self):
        self.locked = not self.locked
        print(f"Auto door zoom target lock: {'LOCKED' if self.locked else 'UNLOCKED'}")

    def reset(self):
        self.locked = False
        self.last_motion_box = None
        self.last_target = None
        self.stable_count = 0
        print("Auto door zoom reset.")

    def detect_motion_box(self, raw_frame):
        frame_h, frame_w = raw_frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None

        scale = 1.0
        work = raw_frame
        if frame_w > AUTO_DOOR_ZOOM_RESIZE_WIDTH:
            scale = AUTO_DOOR_ZOOM_RESIZE_WIDTH / float(frame_w)
            work_h = max(1, int(round(frame_h * scale)))
            work = cv2.resize(raw_frame, (AUTO_DOOR_ZOOM_RESIZE_WIDTH, work_h), interpolation=cv2.INTER_AREA)

        mask = self.subtractor.apply(work)
        _, mask = cv2.threshold(mask, AUTO_DOOR_ZOOM_MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        work_h, work_w = work.shape[:2]
        work_area = float(max(1, work_w * work_h))
        min_area = work_area * AUTO_DOOR_ZOOM_MIN_AREA_RATIO
        max_area = work_area * AUTO_DOOR_ZOOM_MAX_AREA_RATIO

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 15 or h < 20:
                continue
            boxes.append((x, y, x + w, y + h, area))

        if not boxes:
            return None

        boxes.sort(key=lambda item: item[4], reverse=True)
        x1, y1, x2, y2, _ = boxes[0]

        if scale != 1.0:
            inv = 1.0 / scale
            x1 = int(round(x1 * inv))
            y1 = int(round(y1 * inv))
            x2 = int(round(x2 * inv))
            y2 = int(round(y2 * inv))

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        return (x1, y1, x2, y2)

    def update(self, raw_frame, manual_zoom_controller):
        """Return True if the processing zoom crop changed."""
        if not self.enabled or self.locked:
            return False

        frame_h, frame_w = raw_frame.shape[:2]
        motion_box = self.detect_motion_box(raw_frame)
        now = time.time()

        if motion_box is None:
            self.stable_count = max(0, self.stable_count - 1)
            return False

        self.last_motion_box = motion_box

        if self.last_target is None:
            self.stable_count = 1
        else:
            distance = self._center_distance_ratio(motion_box, self.last_target, frame_w, frame_h)
            if distance <= 0.12:
                self.stable_count += 1
            else:
                self.stable_count = 1

        self.last_target = motion_box

        if self.stable_count < AUTO_DOOR_ZOOM_STABLE_FRAMES:
            return False

        if now - self.last_apply_time < AUTO_DOOR_ZOOM_UPDATE_COOLDOWN_SECONDS:
            return False

        center_x, center_y = self._box_center_ratio(motion_box, frame_w, frame_h)
        target_zoom = max(1.0, min(float(MANUAL_ENTRANCE_MAX_ZOOM), float(AUTO_DOOR_ZOOM_TARGET_ZOOM)))
        manual_zoom_controller.set_target(center_x, center_y, zoom=target_zoom, enabled=True)
        manual_zoom_controller.get_crop_params(frame_w, frame_h)

        self.last_apply_time = now

        print(
            "AUTO DOOR ZOOM TARGET FOUND: "
            f"motion_box={motion_box}, zoom={manual_zoom_controller.zoom:.2f}x, "
            f"center=({manual_zoom_controller.center_x_ratio:.3f}, {manual_zoom_controller.center_y_ratio:.3f})"
        )

        if AUTO_DOOR_ZOOM_LOCK_AFTER_FOUND:
            self.locked = True
            print("Auto door zoom locked on this target. Press L to unlock, O to toggle auto mode.")

        return True

    def draw_overlay(self, display_frame, manual_crop_params):
        if not DRAW_AUTO_DOOR_ZOOM_DEBUG:
            return

        frame_h, frame_w = display_frame.shape[:2]
        mode = "ON" if self.enabled else "OFF"
        lock_text = "LOCKED" if self.locked else "SEARCHING"
        text = f"Auto door zoom {mode} | {lock_text} | stable {self.stable_count}/{AUTO_DOOR_ZOOM_STABLE_FRAMES} | O toggle | L lock"
        cv2.putText(
            display_frame,
            text,
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )

        if self.last_motion_box is None:
            return

        if not box_intersects_crop(self.last_motion_box, manual_crop_params):
            return

        x1, y1, x2, y2 = transform_box_to_zoomed_view(
            self.last_motion_box,
            manual_crop_params,
            frame_w,
            frame_h,
        )
        color = (255, 180, 0)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            display_frame,
            "AUTO DOOR/MOTION TARGET",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
        )


# ============================================================
# Common geometry/search helpers
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def make_crop_around_box(frame_w, frame_h, zoom, reference_box):
    crop_w = int(frame_w / zoom)
    crop_h = int(frame_h / zoom)

    cx, cy = box_center(reference_box)

    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))

    left = max(0, min(frame_w - crop_w, left))
    top = max(0, min(frame_h - crop_h, top))

    return left, top, crop_w, crop_h


def make_search_crops(frame_w, frame_h, zoom_levels, full_grid=False, last_box=None):
    """
    Same execution principle as main_fast_attendance.py.

    Fast mode:
      - full frame
      - center crops
      - crops around the last known face

    Full-grid mode:
      - full frame
      - 3x3 grid for each zoom level
    """
    crops = []
    seen = set()

    def add_crop(left, top, crop_w, crop_h, zoom, source):
        left = max(0, min(frame_w - crop_w, int(left)))
        top = max(0, min(frame_h - crop_h, int(top)))

        key = (left, top, crop_w, crop_h)
        if key not in seen:
            seen.add(key)
            crops.append((left, top, crop_w, crop_h, zoom, source))

    for zoom in zoom_levels:
        crop_w = int(frame_w / zoom)
        crop_h = int(frame_h / zoom)

        if zoom == 1.0:
            add_crop(0, 0, frame_w, frame_h, zoom, "full")
            continue

        if last_box is not None:
            left, top, crop_w, crop_h = make_crop_around_box(frame_w, frame_h, zoom, last_box)
            add_crop(left, top, crop_w, crop_h, zoom, f"last-box {zoom:.1f}x")

        add_crop(
            (frame_w - crop_w) // 2,
            (frame_h - crop_h) // 2,
            crop_w,
            crop_h,
            zoom,
            f"center {zoom:.1f}x",
        )

        if full_grid:
            xs = [0, (frame_w - crop_w) // 2, frame_w - crop_w]
            ys = [0, (frame_h - crop_h) // 2, frame_h - crop_h]

            for y in ys:
                for x in xs:
                    add_crop(x, y, crop_w, crop_h, zoom, f"grid {zoom:.1f}x")

    return crops


def resize_for_recognition(image):
    h, w = image.shape[:2]

    if w <= SEARCH_INPUT_WIDTH:
        return image, 1.0, 1.0

    scale = SEARCH_INPUT_WIDTH / float(w)
    new_w = SEARCH_INPUT_WIDTH
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    scale_back_x = w / float(new_w)
    scale_back_y = h / float(new_h)

    return resized, scale_back_x, scale_back_y


def transform_box_to_zoomed_view(box, crop_params, output_w, output_h):
    x1, y1, x2, y2 = box
    left, top, crop_w, crop_h = crop_params

    sx = output_w / crop_w
    sy = output_h / crop_h

    zx1 = int((x1 - left) * sx)
    zy1 = int((y1 - top) * sy)
    zx2 = int((x2 - left) * sx)
    zy2 = int((y2 - top) * sy)

    zx1 = max(0, min(output_w - 1, zx1))
    zy1 = max(0, min(output_h - 1, zy1))
    zx2 = max(0, min(output_w - 1, zx2))
    zy2 = max(0, min(output_h - 1, zy2))

    return zx1, zy1, zx2, zy2


def get_entrance_zone(frame_w, frame_h):
    """
    Returns the door/entrance zone as an absolute pixel box: x1, y1, x2, y2.
    Tune ENTRANCE_ZONE_* in the config section until this rectangle covers the doorway.
    """
    return (
        int(frame_w * ENTRANCE_ZONE_X1),
        int(frame_h * ENTRANCE_ZONE_Y1),
        int(frame_w * ENTRANCE_ZONE_X2),
        int(frame_h * ENTRANCE_ZONE_Y2),
    )


def box_center_inside_zone(box, zone):
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def zone_center_distance(box, zone):
    x1, y1, x2, y2 = box
    zx1, zy1, zx2, zy2 = zone

    face_cx = (x1 + x2) / 2.0
    face_cy = (y1 + y2) / 2.0
    zone_cx = (zx1 + zx2) / 2.0
    zone_cy = (zy1 + zy2) / 2.0

    return ((face_cx - zone_cx) ** 2 + (face_cy - zone_cy) ** 2) ** 0.5



def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_over_union(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_area = box_area_xyxy((ix1, iy1, ix2, iy2))
    if inter_area <= 0:
        return 0.0

    union_area = box_area_xyxy(box_a) + box_area_xyxy(box_b) - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / float(union_area)


def filter_candidates_by_entrance(candidates, frame_w, frame_h):
    if not ENTRANCE_MODE:
        return list(candidates)

    entrance_zone = get_entrance_zone(frame_w, frame_h)
    return [
        c for c in candidates
        if box_center_inside_zone(c["box"], entrance_zone)
    ]


def face_size_xyxy(box):
    x1, y1, x2, y2 = box
    return max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))


def center_distance_ratio(box_a, box_b):
    """
    Distance between centers normalized by face size.
    Useful because IoU alone is not enough when the same face is found
    from different zoom crops and the mapped boxes are slightly shifted.
    """
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    aw, ah = face_size_xyxy(box_a)
    bw, bh = face_size_xyxy(box_b)
    reference = max(aw, ah, bw, bh, 1.0)

    return distance / reference


def candidates_are_same_physical_face(candidate_a, candidate_b, iou_threshold=CANDIDATE_IOU_DEDUPE_THRESHOLD):
    box_a = candidate_a["box"]
    box_b = candidate_b["box"]

    if intersection_over_union(box_a, box_b) >= iou_threshold:
        return True

    # Extra protection for zoom-search duplicates that do not overlap enough
    # but still have almost the same center.
    return center_distance_ratio(box_a, box_b) <= SAME_FACE_CENTER_RATIO


def dedupe_candidates(candidates, iou_threshold=CANDIDATE_IOU_DEDUPE_THRESHOLD):
    """
    Multi-scale search can detect the same real face several times because the
    full frame, center crop, and grid crops overlap. Keep one candidate per
    physical face before updating attendance state.

    Known matches are sorted before Unknown matches, so a red Unknown duplicate
    around a green known face is dropped.
    """
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (
            bool(c.get("is_known", False)),
            float(c.get("similarity", -1.0)),
            float(c.get("area", 0.0)),
        ),
        reverse=True,
    )

    kept = []
    for candidate in sorted_candidates:
        duplicate = False

        for existing in kept:
            if candidates_are_same_physical_face(candidate, existing, iou_threshold=iou_threshold):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


def suppress_unknown_near_known(candidates):
    """
    If a physical face already has a known label, remove nearby Unknown boxes.
    This fixes the common case where the same real face is recognized in one
    crop and classified as Unknown in another crop.
    """
    if not SUPPRESS_UNKNOWN_NEAR_KNOWN:
        return candidates

    known = [c for c in candidates if c.get("is_known", False)]
    if not known:
        return candidates

    filtered = []
    for candidate in candidates:
        if candidate.get("is_known", False):
            filtered.append(candidate)
            continue

        candidate_box = candidate["box"]
        near_known = False
        for known_candidate in known:
            known_box = known_candidate["box"]
            if intersection_over_union(candidate_box, known_box) >= UNKNOWN_SUPPRESSION_IOU:
                near_known = True
                break
            if center_distance_ratio(candidate_box, known_box) <= UNKNOWN_SUPPRESSION_CENTER_RATIO:
                near_known = True
                break

        if not near_known:
            filtered.append(candidate)

    return filtered

def box_intersects_crop(box, crop_params):
    x1, y1, x2, y2 = box
    left, top, crop_w, crop_h = crop_params
    right = left + crop_w
    bottom = top + crop_h

    return not (x2 <= left or x1 >= right or y2 <= top or y1 >= bottom)


def split_display_name(name):
    """Split names like GhinescuLucian, Ghinescu_Lucian, or Ghinescu Lucian."""
    cleaned = str(name).replace("_", " ").replace("-", " ").strip()

    if " " in cleaned:
        return [part for part in cleaned.split() if part]

    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", cleaned)
    return parts or [cleaned]


def short_display_name(name):
    """Return a compact label for the camera view."""
    if not SHORT_LABELS:
        return str(name)

    if not name or str(name) == UNKNOWN_LABEL:
        return UNKNOWN_SHORT_LABEL

    parts = split_display_name(name)

    # For names saved as FamilyNameFirstName, the last CamelCase/token is usually the first name.
    label = parts[-1] if len(parts) >= 2 else parts[0]

    if len(label) > SHORT_LABEL_MAX_CHARS:
        label = label[:SHORT_LABEL_MAX_CHARS]

    return label


# ============================================================
# YuNet/SFace recognition helpers
# ============================================================

def detect_faces(detector, image):
    h, w = image.shape[:2]

    detector.setInputSize((w, h))
    detector.setScoreThreshold(YUNET_SCORE_THRESHOLD)

    _, faces = detector.detect(image)

    if faces is None:
        return np.empty((0, 15), dtype=np.float32)

    return faces


def face_to_xyxy(face):
    x, y, w, h = face[:4]
    return int(x), int(y), int(x + w), int(y + h)


def face_area(face):
    return max(1.0, float(face[2])) * max(1.0, float(face[3]))


def l2_normalize(feature):
    feature = feature.reshape(-1).astype(np.float32)
    norm = np.linalg.norm(feature)

    if norm == 0:
        return feature

    return feature / norm


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def extract_feature(recognizer, image, face):
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return l2_normalize(feature)


def recognize_feature(feature, known_features):
    best_name = UNKNOWN_LABEL
    best_similarity = -1.0

    for person_name, features in known_features.items():
        for known_feature in features:
            similarity = cosine_similarity(feature, known_feature)

            if similarity > best_similarity:
                best_similarity = similarity
                best_name = person_name

    if best_similarity < SFACE_SIMILARITY_THRESHOLD:
        return UNKNOWN_LABEL, best_similarity

    return best_name, best_similarity


def clean_person_name_from_path(image_path: Path, images_root: Path):
    """
    Supports:
      images/GhinescuLucian.jpg     -> GhinescuLucian
      images/IuliaSocarde2.jpg      -> IuliaSocarde
      images/Lucian/1.jpg           -> Lucian
    """
    relative = image_path.relative_to(images_root)

    if len(relative.parts) >= 2:
        return relative.parts[0]

    name = image_path.stem.strip()
    name = re.sub(r"[\s_-]*\d+$", "", name).strip()

    return name


def find_image_files(images_root: Path):
    return sorted(
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_known_features(detector, recognizer, images_dir):
    images_root = Path(images_dir)

    if not images_root.exists():
        raise RuntimeError(f"Images folder does not exist: {images_root.resolve()}")

    image_files = find_image_files(images_root)

    if not image_files:
        raise RuntimeError(f"No image files found in: {images_root.resolve()}")

    known_features: Dict[str, List[np.ndarray]] = {}

    print(f"Loading known faces from: {images_root.resolve()}")
    print(f"Found {len(image_files)} image files.\n")

    for image_path in image_files:
        person_name = clean_person_name_from_path(image_path, images_root)
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] Could not read image: {image_path.name}")
            continue

        faces = detect_faces(detector, image)

        if len(faces) == 0:
            print(f"[SKIP] No face detected in: {image_path.name}")
            continue

        largest = max(faces, key=face_area)
        feature = extract_feature(recognizer, image, largest)

        known_features.setdefault(person_name, []).append(feature)

        print(f"[OK] {image_path.name} -> {person_name}")

    if not known_features:
        raise RuntimeError("No valid face features were loaded.")

    print("\nLoaded people:")
    for name, features in known_features.items():
        print(f"  {name}: {len(features)} feature(s)")
    print()

    return known_features


# ============================================================
# Embedding-file loading helpers
# ============================================================

def clean_person_name_from_embedding_file(file_path: Path):
    """
    Examples:
      images/GhinescuLucian_embeddings_1.json -> GhinescuLucian
      images/GhinescuLucian_embeddings_1      -> GhinescuLucian
      images/embeddings_1.json                -> embeddings_1
    """
    name = file_path.stem.strip()
    name = re.sub(r"[\s_-]*embeddings?[\s_-]*\d*$", "", name, flags=re.IGNORECASE).strip()
    name = re.sub(r"[\s_-]*\d+$", "", name).strip()
    return name or file_path.stem.strip()


def is_number(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def is_embedding_vector(value):
    return isinstance(value, list) and len(value) >= 64 and all(is_number(x) for x in value)


def is_embedding_matrix(value):
    return isinstance(value, list) and len(value) > 0 and all(is_embedding_vector(row) for row in value)


def add_embedding(known_features: Dict[str, List[np.ndarray]], person_name: str, embedding, source_name: str):
    if not person_name:
        raise RuntimeError(f"Missing person name for embedding in {source_name}")

    feature = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if not np.all(np.isfinite(feature)):
        print(f"[SKIP] Invalid numbers in embedding for {person_name} from {source_name}")
        return

    # SFace normally produces 128-float features. Warn, but do not crash,
    # because your Android side may still serialize the same vector in a custom way.
    if feature.size != 128:
        print(f"[WARN] {source_name}: embedding for {person_name} has length {feature.size}, expected 128 for SFace")

    known_features.setdefault(person_name, []).append(l2_normalize(feature))


def parse_embedding_json(data, known_features: Dict[str, List[np.ndarray]], fallback_name: str, source_name: str):
    """
    Accepted JSON formats:

    1) One person per file:
       {"name": "GhinescuLucian", "embedding": [0.1, ...]}
       {"name": "GhinescuLucian", "embeddings": [[0.1, ...], [0.2, ...]]}

    2) Phone/web enrollment file:
       {
         "name": "GhinescuLucian",
         "model": {"family": "opencv", "recognizer": "sface"},
         "captures": [
           {"pose": "front", "label": "Front", "descriptor": [0.1, ...]},
           {"pose": "right", "label": "Head right", "descriptor": [0.2, ...]}
         ]
       }

       IMPORTANT: in this format, every descriptor under captures belongs to the
       TOP-LEVEL person name. The capture label is only a pose label, not a person.

    3) File name contains person name, JSON is only the vector/matrix:
       [0.1, 0.2, ...]
       [[0.1, ...], [0.2, ...]]

    4) One combined database file:
       {"GhinescuLucian": [[0.1, ...], [0.2, ...]], "IuliaSocarde": [[...]]}

    5) List of records:
       [{"name": "GhinescuLucian", "embedding": [0.1, ...]}]
    """
    # Do NOT treat "label" as a person name inside phone captures.
    # In your phone JSON, label = "Front", "Head right", etc.
    primary_name_keys = ("name", "person", "person_name", "personName", "student", "studentName")
    fallback_id_keys = ("studentId", "student_id", "id")
    embedding_keys = ("embedding", "embeddings", "feature", "features", "vector", "vectors", "descriptor", "descriptors")
    container_keys = ("people", "persons", "students", "records", "items", "data")

    if is_embedding_vector(data):
        add_embedding(known_features, fallback_name, data, source_name)
        return

    if is_embedding_matrix(data):
        for embedding in data:
            add_embedding(known_features, fallback_name, embedding, source_name)
        return

    if isinstance(data, list):
        for item in data:
            parse_embedding_json(item, known_features, fallback_name, source_name)
        return

    if isinstance(data, dict):
        person_name = None
        for key in primary_name_keys:
            if key in data and data[key]:
                person_name = str(data[key]).strip()
                break

        if not person_name:
            for key in fallback_id_keys:
                if key in data and data[key]:
                    person_name = str(data[key]).strip()
                    break

        if not person_name:
            person_name = fallback_name

        # Special handling for the phone/web enrollment JSON.
        # The top-level name is the student/person. Each capture label is only a pose.
        if "captures" in data and isinstance(data["captures"], list):
            model = data.get("model", {})
            family = str(model.get("family", "")).lower() if isinstance(model, dict) else ""
            recognizer = str(model.get("recognizer", "")).lower() if isinstance(model, dict) else ""

            if family and family != "opencv":
                print(f"[WARN] {source_name}: model.family is '{family}', expected 'opencv' for SFace embeddings")
            if recognizer and recognizer != "sface":
                print(f"[WARN] {source_name}: model.recognizer is '{recognizer}', expected 'sface'")

            loaded_any = False

            for capture in data["captures"]:
                if not isinstance(capture, dict):
                    continue

                pose = str(capture.get("pose") or capture.get("label") or "capture")

                for key in embedding_keys:
                    if key not in capture:
                        continue

                    value = capture[key]

                    if is_embedding_vector(value):
                        add_embedding(known_features, person_name, value, f"{source_name} / {pose}")
                        loaded_any = True
                        break

                    if is_embedding_matrix(value):
                        for embedding in value:
                            add_embedding(known_features, person_name, embedding, f"{source_name} / {pose}")
                            loaded_any = True
                        break

            if loaded_any:
                return

        for container_key in container_keys:
            if container_key in data:
                parse_embedding_json(data[container_key], known_features, person_name, source_name)
                return

        for key in embedding_keys:
            if key in data:
                value = data[key]
                if is_embedding_vector(value):
                    add_embedding(known_features, person_name, value, source_name)
                    return
                if is_embedding_matrix(value):
                    for embedding in value:
                        add_embedding(known_features, person_name, embedding, source_name)
                    return

        # Treat remaining dicts as: {"PersonName": vector_or_matrix, ...}
        # Avoid descending into metadata such as model/version/createdAt.
        metadata_keys = {
            "version", "createdAt", "created_at", "model", "quality", "capturedAt",
            "captured_at", "pose", "label", "detector", "recognizer", "note",
            "studentId", "student_id", "name"
        }

        loaded_any = False
        for key, value in data.items():
            if key in metadata_keys:
                continue

            if is_embedding_vector(value):
                add_embedding(known_features, str(key), value, source_name)
                loaded_any = True
            elif is_embedding_matrix(value):
                for embedding in value:
                    add_embedding(known_features, str(key), embedding, source_name)
                    loaded_any = True
            elif isinstance(value, dict) or isinstance(value, list):
                before = sum(len(v) for v in known_features.values())
                parse_embedding_json(value, known_features, person_name, source_name)
                after = sum(len(v) for v in known_features.values())
                loaded_any = loaded_any or after > before

        if loaded_any:
            return

    print(f"[SKIP] Could not understand embedding JSON format in {source_name}")

def find_embedding_files(source: Path):
    if source.is_file():
        return [source]

    candidates = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.lower()
        suffix = path.suffix.lower()

        if "embedding" in lower_name or suffix == ".json":
            candidates.append(path)

    return sorted(set(candidates))


def load_known_features_from_embeddings(embeddings_source):
    source = Path(embeddings_source)

    if not source.exists():
        raise RuntimeError(f"Embeddings source does not exist: {source.resolve()}")

    embedding_files = find_embedding_files(source)

    if not embedding_files:
        raise RuntimeError(f"No embedding JSON files found in: {source.resolve()}")

    known_features: Dict[str, List[np.ndarray]] = {}

    print(f"Loading known face embeddings from: {source.resolve()}")
    print(f"Found {len(embedding_files)} embedding file(s).\n")

    for embedding_path in embedding_files:
        fallback_name = clean_person_name_from_embedding_file(embedding_path)

        try:
            with open(embedding_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[SKIP] Could not read JSON from {embedding_path.name}: {exc}")
            continue

        before = sum(len(v) for v in known_features.values())
        parse_embedding_json(data, known_features, fallback_name, embedding_path.name)
        after = sum(len(v) for v in known_features.values())

        if after > before:
            print(f"[OK] {embedding_path.name}: loaded {after - before} embedding(s)")

    if not known_features:
        raise RuntimeError("No valid embeddings were loaded.")

    print("\nLoaded people from embeddings:")
    for name, features in known_features.items():
        print(f"  {name}: {len(features)} embedding(s)")
    print()

    return known_features


def map_detection_box_to_original(face, left, top, base_to_original_x, base_to_original_y, det_to_base_x, det_to_base_y):
    dx1, dy1, dx2, dy2 = face_to_xyxy(face)

    bx1 = dx1 * det_to_base_x
    by1 = dy1 * det_to_base_y
    bx2 = dx2 * det_to_base_x
    by2 = dy2 * det_to_base_y

    ox1 = int(left + bx1 * base_to_original_x)
    oy1 = int(top + by1 * base_to_original_y)
    ox2 = int(left + bx2 * base_to_original_x)
    oy2 = int(top + by2 * base_to_original_y)

    return ox1, oy1, ox2, oy2


def detect_faces_with_search_zoom(detector, recognizer, known_features, frame, full_grid=False, last_box=None, locked_name=None):
    """
    Same principle as the SimpleFacerec version, but with YuNet/SFace:
      - crop the frame
      - enlarge zoomed crops
      - detect face with YuNet
      - extract SFace feature
      - compare with known features
      - map detected boxes back to original frame coordinates
    """
    frame_h, frame_w = frame.shape[:2]
    candidates = []

    zoom_levels = SEARCH_ZOOM_LEVELS_FULL if full_grid else SEARCH_ZOOM_LEVELS_FAST

    search_crops = make_search_crops(
        frame_w,
        frame_h,
        zoom_levels=zoom_levels,
        full_grid=full_grid,
        last_box=last_box,
    )

    for left, top, crop_w, crop_h, search_zoom, source in search_crops:
        crop = frame[top:top + crop_h, left:left + crop_w]

        if search_zoom == 1.0:
            base_image = crop
            base_to_original_x = 1.0
            base_to_original_y = 1.0
        else:
            # Enlarge crop so a far face becomes easier to detect/recognize.
            base_image = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            base_to_original_x = crop_w / float(frame_w)
            base_to_original_y = crop_h / float(frame_h)

        recognition_input, det_to_base_x, det_to_base_y = resize_for_recognition(base_image)

        faces = detect_faces(detector, recognition_input)

        if len(faces) == 0:
            continue

        faces = sorted(faces, key=face_area, reverse=True)

        for face in faces:
            feature = extract_feature(recognizer, recognition_input, face)
            name, similarity = recognize_feature(feature, known_features)

            original_box = map_detection_box_to_original(
                face=face,
                left=left,
                top=top,
                base_to_original_x=base_to_original_x,
                base_to_original_y=base_to_original_y,
                det_to_base_x=det_to_base_x,
                det_to_base_y=det_to_base_y,
            )

            x1, y1, x2, y2 = original_box
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            area = box_w * box_h
            is_known = bool(name) and name != UNKNOWN_LABEL

            candidates.append({
                "box": original_box,
                "name": name,
                "is_known": is_known,
                "similarity": similarity,
                "area": area,
                "source": source,
                "search_zoom": search_zoom,
            })

        # Do not return early when a known face is found. In multi-face mode we
        # must continue scanning the remaining crops, otherwise the first known
        # face dominates and farther faces are never recognized.

    return candidates


def choose_target_face(candidates, frame_w, frame_h, locked_name=None, last_box=None):
    """
    Entrance-aware target selection.

    In normal mode it behaves like the old code.
    In ENTRANCE_MODE, it only chooses a face whose center is inside the door zone.
    This makes the camera behave like a classroom entrance queue: follow the first/front
    person in the doorway, mark that person, then release and look for the next one.
    """
    if not candidates:
        return None

    frame_center_box = (frame_w // 2, frame_h // 2, frame_w // 2 + 1, frame_h // 2 + 1)

    if ENTRANCE_MODE:
        entrance_zone = get_entrance_zone(frame_w, frame_h)
        candidates = [
            c for c in candidates
            if box_center_inside_zone(c["box"], entrance_zone)
        ]

        if not candidates:
            return None

    if locked_name:
        same_name = [c for c in candidates if c["name"] == locked_name]

        if same_name:
            reference = last_box if last_box is not None else frame_center_box
            same_name.sort(key=lambda c: center_distance(c["box"], reference))
            return same_name[0]

        if last_box is not None:
            candidates.sort(key=lambda c: center_distance(c["box"], last_box))
            return candidates[0]

    known_faces = [c for c in candidates if c["is_known"]]

    if known_faces:
        if ENTRANCE_MODE and ENTRANCE_SORT_BY_BIGGEST_FACE:
            entrance_zone = get_entrance_zone(frame_w, frame_h)
            known_faces.sort(
                key=lambda c: (
                    -c["area"],
                    -c["similarity"],
                    zone_center_distance(c["box"], entrance_zone),
                )
            )
        else:
            known_faces.sort(
                key=lambda c: (
                    -c["similarity"],
                    -c["area"],
                    center_distance(c["box"], frame_center_box),
                )
            )

        return known_faces[0]

    # If no known face is recognized yet, still follow the largest face in the zone.
    if ENTRANCE_MODE:
        entrance_zone = get_entrance_zone(frame_w, frame_h)
        candidates.sort(key=lambda c: (-c["area"], zone_center_distance(c["box"], entrance_zone)))
    else:
        candidates.sort(key=lambda c: (-c["area"], center_distance(c["box"], frame_center_box)))

    return candidates[0]


def make_zoomed_proof_frame(frame, face_box):
    """Create a proof photo centered on the specific face being marked."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = face_box
    face_h = max(1, y2 - y1)

    proof_zoom = (FACE_TARGET_HEIGHT * frame_h) / face_h
    proof_zoom = max(MIN_ZOOM, min(MAX_ZOOM, proof_zoom))

    crop_params = make_crop_around_box(frame_w, frame_h, proof_zoom, face_box)
    return FaceCenterZoomer.apply_zoom(frame, crop_params)


# ============================================================
# Attendance/photo helpers
# ============================================================

def save_person_photo(name, zoomed_frame):
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}_person.jpg")
    cv2.imwrite(image_path, zoomed_frame)

    return image_path


def mark_attendance(name, photo_path):
    file_exists = os.path.exists(ATTENDANCE_CSV)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "name", "photo_path"])

        writer.writerow([timestamp, name, photo_path])

    return timestamp


def faceattendance_externalid_for_name(name):
    """Translate a local recognition name to the Moodle FaceAttendance externalid."""
    return FACE_EXTERNAL_ID_BY_NAME.get(str(name), str(name))


def faceattendance_ssl_context():
    if FACE_ATTENDANCE_VERIFY_TLS:
        return None

    return ssl._create_unverified_context()


def faceattendance_http_json(url, method="GET", payload=None):
    """Small JSON HTTP helper that avoids adding a requests dependency."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "faceattendance-python-yunet-sface/1.0",
    }

    if FACE_ATTENDANCE_SECRET:
        headers["X-FaceAttendance-Secret"] = FACE_ATTENDANCE_SECRET

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=FACE_ATTENDANCE_TIMEOUT_SECONDS,
            context=faceattendance_ssl_context(),
        ) as response:
            status = getattr(response, "status", response.getcode())
            raw_body = response.read().decode("utf-8", errors="replace")

        parsed_body = None
        if raw_body.strip():
            try:
                parsed_body = json.loads(raw_body)
            except json.JSONDecodeError:
                parsed_body = raw_body

        return {
            "ok": 200 <= status < 300,
            "status": status,
            "body": parsed_body,
            "raw_body": raw_body,
            "error": None,
        }

    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_body = json.loads(raw_body) if raw_body.strip() else None
        except json.JSONDecodeError:
            parsed_body = raw_body

        return {
            "ok": False,
            "status": exc.code,
            "body": parsed_body,
            "raw_body": raw_body,
            "error": str(exc),
        }

    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "body": None,
            "raw_body": "",
            "error": str(exc),
        }


def collect_roster_externalids(data):
    """Extract externalid values from whatever JSON shape the roster endpoint returns."""
    externalids = set()

    if isinstance(data, dict):
        for key, value in data.items():
            if key in {"externalid", "external_id", "face_externalid"} and value:
                externalids.add(str(value))
            else:
                externalids.update(collect_roster_externalids(value))

    elif isinstance(data, list):
        for item in data:
            externalids.update(collect_roster_externalids(item))

    return externalids


def fetch_faceattendance_roster():
    if not FACE_ATTENDANCE_MARK_ENABLED or not FACE_ATTENDANCE_ROSTER_URL:
        return None

    result = faceattendance_http_json(FACE_ATTENDANCE_ROSTER_URL, method="GET")

    if not result["ok"]:
        print(
            "[WARN] Could not read Moodle roster. "
            f"status={result['status']} error={result['error']} body={result['raw_body'][:300]}"
        )
        return None

    externalids = collect_roster_externalids(result["body"])
    print(f"Moodle roster loaded: {len(externalids)} externalid value(s) found.")
    return externalids


def mark_faceattendance_remote(name, confidence):
    """POST one detection to Moodle FaceAttendance mark.php."""
    externalid = faceattendance_externalid_for_name(name)

    try:
        confidence_value = float(confidence)
    except Exception:
        confidence_value = 0.0

    # The API expects a confidence-like value. Keep it inside a normal 0..1 range.
    confidence_value = max(0.0, min(1.0, confidence_value))

    payload = {
        "cmid": FACE_ATTENDANCE_CMID,
        "secret": FACE_ATTENDANCE_SECRET,
        "detections": [
            {
                "externalid": externalid,
                "confidence": round(confidence_value, 4),
                "source": FACE_ATTENDANCE_SOURCE,
            }
        ],
    }

    result = faceattendance_http_json(
        FACE_ATTENDANCE_MARK_URL,
        method="POST",
        payload=payload,
    )
    result["externalid"] = externalid
    result["payload"] = payload
    return result


def mark_faceattendance_remote_and_print(name, confidence):
    result = mark_faceattendance_remote(name, confidence)
    externalid = result.get("externalid", faceattendance_externalid_for_name(name))

    if result["ok"]:
        print(f"MOODLE ATTENDANCE MARKED: name={name} externalid={externalid} status={result['status']}")
    else:
        print(
            "MOODLE ATTENDANCE FAILED: "
            f"name={name} externalid={externalid} status={result['status']} "
            f"error={result['error']} body={result['raw_body'][:300]}"
        )

    return result


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ============================================================
# Main execution path: mirrors main_fast_attendance.py
# ============================================================

def main():
    global DRAW_UNKNOWN_FACES, ENABLE_PERIODIC_GRID_SEARCH

    check_opencv_api()
    ensure_model_file(YUNET_MODEL, YUNET_URL)
    ensure_model_file(SFACE_MODEL, SFACE_URL)

    detector = create_detector()
    recognizer = create_recognizer()

    known_features = load_known_features_from_embeddings(EMBEDDINGS_SOURCE)

    print(f"Moodle mark endpoint: {FACE_ATTENDANCE_MARK_URL}")
    print(f"Moodle roster endpoint: {FACE_ATTENDANCE_ROSTER_URL}")
    print(f"Moodle course module id: {FACE_ATTENDANCE_CMID}")
    print(f"Moodle source: {FACE_ATTENDANCE_SOURCE}")
    if FACE_ATTENDANCE_SECRET == "your-api-secret-here":
        print("[WARN] FACE_ATTENDANCE_SECRET is still the placeholder value. Replace it with the plugin secret.")
    if FACE_ATTENDANCE_MARK_ENABLED and not FACE_ATTENDANCE_VERIFY_TLS:
        print("[WARN] TLS certificate verification is disabled for local HTTPS testing. Enable it for production.")

    roster_externalids = fetch_faceattendance_roster()

    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"YuNet score threshold: {YUNET_SCORE_THRESHOLD}")
    print(f"SFace similarity threshold: {SFACE_SIMILARITY_THRESHOLD}")
    print("Press ESC to exit. Press R to reset tracking state.")
    print("Fast mode: full-frame detection, unknown faces enabled, periodic grid search OFF by default.")
    print("Press U to toggle Unknown drawing. Press G to toggle slower grid/zoom search.")
    print('Manual entrance zoom controls: visible sliders in "Manual Zoom Controls" window, plus +/- zoom, WASD/arrows pan, M toggle, Z reset, P print .env values.')
    print("Click inside the camera window first if the keyboard controls do not react.")
    print("Manual adjustment now takes control: pressing +/-/WASD/arrows/Z/M disables auto-door zoom until you press O again.")
    print("Auto door zoom controls: O toggle auto-door search, L lock/unlock current auto target.\n")

    # Per-person state. Every recognized person gets an independent stability counter.
    face_state_by_name = defaultdict(lambda: {
        "stable_count": 0,
        "last_seen": 0.0,
        "last_box": None,
        "last_similarity": -1.0,
    })

    last_capture_time_by_name = defaultdict(lambda: 0.0)
    saved_count_by_name = defaultdict(int)
    last_attendance_time_by_name = defaultdict(lambda: 0.0)

    frame_index = 0
    last_candidates = []

    manual_zoom = ManualEntranceZoomController(actual_w, actual_h)
    manual_zoom.create_control_panel()
    auto_door_zoom = AutoDoorZoomController()
    print(
        "Manual entrance zoom initial state: "
        f"{'ON' if manual_zoom.enabled else 'OFF'}, "
        f"zoom={manual_zoom.zoom:.2f}x, "
        f"center=({manual_zoom.center_x_ratio:.3f}, {manual_zoom.center_y_ratio:.3f})"
    )
    print(
        "Auto door zoom initial state: "
        f"{'ON' if auto_door_zoom.enabled else 'OFF'}, "
        f"target_zoom={AUTO_DOOR_ZOOM_TARGET_ZOOM:.2f}x, "
        f"lock_after_found={'true' if AUTO_DOOR_ZOOM_LOCK_AFTER_FOUND else 'false'}"
    )

    moodle_executor = ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read frame from camera.")
                break

            now = time.time()
            frame_index += 1

            # Keep the raw full camera frame for automatic door/opening search.
            # The actual face detection still runs on the current processing view below.
            raw_frame = frame.copy()

            # Visible slider controls have priority over auto-door zoom.
            # If you move any slider, auto-door is disabled so it cannot fight your manual entrance framing.
            if manual_zoom.update_from_control_panel(raw_frame.shape[1], raw_frame.shape[0]):
                last_candidates = []
                if auto_door_zoom.enabled:
                    auto_door_zoom.enabled = False
                    auto_door_zoom.locked = False
                    print("Auto door zoom disabled because you used the visible manual zoom sliders. Press O to enable auto-door zoom again.")

            if auto_door_zoom.update(raw_frame, manual_zoom):
                manual_zoom.sync_control_panel_to_state()
                last_candidates = []

            # Apply manual/auto digital zoom before detection. From this point onward,
            # all boxes/candidates/photo proofs are in the zoomed processing view.
            frame, manual_crop_params = manual_zoom.apply_to_frame(frame)
            frame_h, frame_w = frame.shape[:2]

            should_fast_search = frame_index % FAST_SEARCH_EVERY_N_FRAMES == 0
            should_full_grid_search = ENABLE_PERIODIC_GRID_SEARCH and frame_index % FULL_GRID_SEARCH_EVERY_N_FRAMES == 0

            candidates = last_candidates

            if should_fast_search or should_full_grid_search:
                candidates = detect_faces_with_search_zoom(
                    detector=detector,
                    recognizer=recognizer,
                    known_features=known_features,
                    frame=frame,
                    full_grid=should_full_grid_search,
                    last_box=None,
                    locked_name=None,
                )

                # Do NOT filter by the entrance zone here. The goal is maximum faces.
                # Multi-scale/grid crops can detect the same physical face several times,
                # so dedupe after collecting all detections.
                candidates = dedupe_candidates(candidates)
                candidates = suppress_unknown_near_known(candidates)
                last_candidates = candidates

            # Display the current processing view. If manual entrance zoom is ON,
            # this is the zoomed entrance crop, not the raw full camera frame.
            display_frame = frame.copy()

            # Update stability and attendance for every known face independently.
            seen_known_names = set()
            for candidate in candidates:
                if not candidate.get("is_known", False):
                    continue

                name = candidate["name"]

                # If the same person is found twice after dedupe, keep only the best one for state.
                if name in seen_known_names:
                    continue
                seen_known_names.add(name)

                state = face_state_by_name[name]
                state["stable_count"] += 1
                state["last_seen"] = now
                state["last_box"] = candidate["box"]
                state["last_similarity"] = candidate.get("similarity", -1.0)

                if state["stable_count"] >= STABLE_FRAMES_REQUIRED and SAVE_PHOTO_WHEN_KNOWN_STABLE:
                    attendance_cooldown_ok = (
                        now - last_attendance_time_by_name[name] >= ATTENDANCE_COOLDOWN_SECONDS
                    )
                    photo_limit_ok = saved_count_by_name[name] < MAX_PHOTOS_PER_PERSON
                    photo_cooldown_ok = now - last_capture_time_by_name[name] >= PHOTO_COOLDOWN_SECONDS

                    if attendance_cooldown_ok and photo_limit_ok and photo_cooldown_ok:
                        clean_zoomed = make_zoomed_proof_frame(frame, candidate["box"])
                        image_path = save_person_photo(name, clean_zoomed)
                        timestamp = mark_attendance(name, image_path)

                        last_capture_time_by_name[name] = now
                        last_attendance_time_by_name[name] = now
                        saved_count_by_name[name] += 1

                        print(f"ATTENDANCE MARKED LOCALLY: {name} at {timestamp}")
                        print(f"Saved proof photo: {image_path}")

                        externalid = faceattendance_externalid_for_name(name)
                        if roster_externalids is not None and externalid not in roster_externalids:
                            print(
                                f"[WARN] Recognized name '{name}' maps to externalid '{externalid}', "
                                "but this value was not found in the Moodle roster response."
                            )

                        if FACE_ATTENDANCE_MARK_ENABLED:
                            moodle_executor.submit(
                                mark_faceattendance_remote_and_print,
                                name,
                                candidate.get("similarity", 0.0),
                            )

            # Remove disappeared people so their stability must be rebuilt if they re-enter.
            for name in list(face_state_by_name.keys()):
                if now - face_state_by_name[name]["last_seen"] > FACE_STATE_TIMEOUT_SECONDS:
                    del face_state_by_name[name]

            # Draw every detected face directly on the original full frame.
            unknown_draw_index = 0
            for candidate in candidates:
                is_known = candidate.get("is_known", False)

                if is_known and not DRAW_ALL_RECOGNIZED_FACES:
                    continue
                if not is_known and not DRAW_UNKNOWN_FACES:
                    continue

                x1, y1, x2, y2 = candidate["box"]
                x1 = max(0, min(frame_w - 1, int(x1)))
                y1 = max(0, min(frame_h - 1, int(y1)))
                x2 = max(0, min(frame_w - 1, int(x2)))
                y2 = max(0, min(frame_h - 1, int(y2)))

                name = candidate.get("name", UNKNOWN_LABEL)
                color = (0, 255, 0) if is_known else (0, 0, 255)

                if is_known:
                    label = short_display_name(name)
                    if DRAW_SIMILARITY_ON_LABEL and "similarity" in candidate:
                        label += f" {candidate['similarity']:.2f}"
                    if DRAW_STABILITY_ON_LABEL:
                        stable_count = face_state_by_name.get(name, {}).get("stable_count", 0)
                        label += f" {min(stable_count, STABLE_FRAMES_REQUIRED)}/{STABLE_FRAMES_REQUIRED}"
                else:
                    unknown_draw_index += 1
                    label = f"{UNKNOWN_SHORT_LABEL}{unknown_draw_index}"
                    if DRAW_SIMILARITY_ON_LABEL and "similarity" in candidate:
                        label += f" {candidate['similarity']:.2f}"

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.58,
                    color,
                    2,
                )

                if DRAW_DEBUG_SEARCH_SOURCE:
                    cv2.putText(
                        display_frame,
                        f"source: {candidate.get('source', '')}",
                        (x1, min(frame_h - 15, y2 + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        2,
                    )

            known_count = sum(1 for c in candidates if c.get("is_known", False))
            unknown_count = len(candidates) - known_count
            status = f"Detected {len(candidates)} face(s) | Known {known_count} | Unknown {unknown_count}"
            if not DRAW_UNKNOWN_FACES:
                status += " | Unknown hidden"

            if should_full_grid_search:
                status += " | grid search"
            elif should_fast_search:
                status += " | fast full-frame"
            else:
                status += " | cached"

            if not ENABLE_PERIODIC_GRID_SEARCH:
                status += " | grid OFF"

            cv2.putText(
                display_frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                "ESC | sliders: Manual Zoom Controls | M manual | +/- zoom | WASD/arrows pan | O auto",
                (20, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                2,
            )

            manual_zoom.draw_overlay(display_frame, manual_crop_params)
            auto_door_zoom.draw_overlay(display_frame, manual_crop_params)

            cv2.imshow(WINDOW_NAME, display_frame)

            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF

            if key == 27:
                break

            if key in (ord("r"), ord("R")):
                face_state_by_name.clear()
                last_candidates = []
                print("Reset tracking state. Attendance/photo counters were kept.")

            if key in (ord("u"), ord("U")):
                DRAW_UNKNOWN_FACES = not DRAW_UNKNOWN_FACES
                state = "ON" if DRAW_UNKNOWN_FACES else "OFF"
                print(f"Unknown face drawing: {state}")

            if key in (ord("g"), ord("G")):
                ENABLE_PERIODIC_GRID_SEARCH = not ENABLE_PERIODIC_GRID_SEARCH
                state = "ON" if ENABLE_PERIODIC_GRID_SEARCH else "OFF"
                last_candidates = []
                print(f"Periodic grid/zoom search: {state}")

            if key in (ord("o"), ord("O")):
                auto_door_zoom.toggle()

            if key in (ord("l"), ord("L")):
                auto_door_zoom.toggle_lock()

            if manual_zoom.handle_key(key_raw, frame_w, frame_h):
                last_candidates = []
                if auto_door_zoom.enabled:
                    auto_door_zoom.enabled = False
                    auto_door_zoom.locked = False
                    print("Auto door zoom disabled because you manually adjusted the zoom. Press O to enable auto-door zoom again.")

    finally:
        moodle_executor.shutdown(wait=False)
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
