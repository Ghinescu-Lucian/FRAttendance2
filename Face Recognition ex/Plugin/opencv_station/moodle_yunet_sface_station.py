import os
import re
import time
import csv
import json
import base64
import urllib.request
import urllib.parse
import urllib.error
import ssl
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass


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
# Run as Moodle camera station:
#   py moodle_yunet_sface_station.py
#
# Configure MOODLE_BASE_URL, MOODLE_CMID, and MOODLE_API_SECRET below before running.
# ============================================================


# =========================
# Configuration
# =========================
CAMERA_INDEX = int(os.environ.get("FACEATTENDANCE_CAMERA_INDEX", "0"))
# Folder or file containing precomputed embeddings.
# You can point this to:
#   - one file:   "images/GhinescuLucian_embeddings_1.json"
#   - a folder:   "images/"
# The loader accepts files named like *_embeddings*, embeddings_*.json, or normal .json files.
EMBEDDINGS_SOURCE = "images/"

# Moodle integration.
# Keep USE_MOODLE_API = os.environ.get("FACEATTENDANCE_USE_MOODLE_API", "true").lower() not in ("0", "false", "no", "off") when this script is used as the classroom camera station.
# The script will download embeddings from Moodle and POST attendance/unknown detections back to Moodle.
USE_MOODLE_API = os.environ.get("FACEATTENDANCE_USE_MOODLE_API", "true").lower() not in ("0", "false", "no", "off")
MOODLE_BASE_URL = os.environ.get("FACEATTENDANCE_MOODLE_BASE_URL", "http://localhost/moodle")
MOODLE_CMID = int(os.environ.get("FACEATTENDANCE_CMID", "12"))
MOODLE_API_SECRET = os.environ.get("FACEATTENDANCE_API_SECRET", "change-this-secret")
MOODLE_STATION_SOURCE = os.environ.get("FACEATTENDANCE_SOURCE", "opencv-yunet-sface-station")
MOODLE_REFRESH_SECONDS = int(float(os.environ.get("FACEATTENDANCE_REFRESH_SECONDS", "30")))
UNKNOWN_MOODLE_COOLDOWN_SECONDS = int(float(os.environ.get("FACEATTENDANCE_UNKNOWN_COOLDOWN_SECONDS", "15")))
MOODLE_HTTP_TIMEOUT_SECONDS = int(float(os.environ.get("FACEATTENDANCE_HTTP_TIMEOUT_SECONDS", "10")))
MOODLE_VERIFY_TLS = os.environ.get("FACEATTENDANCE_VERIFY_TLS", "true").lower() not in ("0", "false", "no", "off")
UNKNOWN_THUMBNAIL_JPEG_QUALITY = int(float(os.environ.get("FACEATTENDANCE_THUMBNAIL_QUALITY", "80")))
UNKNOWN_THUMBNAIL_MAX_WIDTH = int(float(os.environ.get("FACEATTENDANCE_THUMBNAIL_MAX_WIDTH", "360")))
UNKNOWN_THUMBNAIL_PADDING_RATIO = 0.35

# Manual digital zoom for the local OpenCV station.
# This zoom is applied before face detection, so it helps the detector/recognizer
# focus on a classroom entrance/door area and makes far faces larger.
# Controls at runtime:
#   M       toggle manual zoom on/off
#   + or =  zoom in
#   - or _  zoom out
#   0       reset zoom to full frame
#   W/A/S/D pan the zoomed region up/left/down/right
MANUAL_ZOOM_ENABLED = os.environ.get("FACEATTENDANCE_MANUAL_ZOOM_ENABLED", "false").lower() in ("1", "true", "yes", "on")
MANUAL_ZOOM_FACTOR = float(os.environ.get("FACEATTENDANCE_MANUAL_ZOOM", "1.0"))
MANUAL_ZOOM_MIN = 1.0
MANUAL_ZOOM_MAX = float(os.environ.get("FACEATTENDANCE_MANUAL_ZOOM_MAX", "5.0"))
MANUAL_ZOOM_CENTER_X = float(os.environ.get("FACEATTENDANCE_MANUAL_ZOOM_CENTER_X", "0.5"))
MANUAL_ZOOM_CENTER_Y = float(os.environ.get("FACEATTENDANCE_MANUAL_ZOOM_CENTER_Y", "0.5"))
MANUAL_ZOOM_STEP = float(os.environ.get("FACEATTENDANCE_MANUAL_ZOOM_STEP", "0.25"))
MANUAL_ZOOM_PAN_STEP = float(os.environ.get("FACEATTENDANCE_MANUAL_ZOOM_PAN_STEP", "0.06"))

# Runtime Moodle state. Filled by load_known_features_from_moodle().
MOODLE_PERSON_LABELS = {}
MOODLE_ACTIVE_SESSION = None
MOODLE_LAST_BOOTSTRAP = 0.0


# Recognition algorithm profile. This can be overridden by Moodle's activity setting
# returned by api/station_bootstrap.php or locally through environment variable
# FACEATTENDANCE_ALGORITHM_PROFILE.
ALGORITHM_PROFILE = os.environ.get("FACEATTENDANCE_ALGORITHM_PROFILE", "fast_short")
# Optional: send a selected camera-station profile to Moodle bootstrap.
# Leave empty to use the local ALGORITHM_PROFILE value.
MOODLE_ALGORITHM_PROFILE = os.environ.get("FACEATTENDANCE_STATION_PROFILE", ALGORITHM_PROFILE)
ACTIVE_ALGORITHM_PROFILE = None

ALGORITHM_PROFILES = {
    # Based on main_yunet_sface_many_faces_unknown_fast_short.py.
    "fast_short": {
        "label": "Fast short labels",
        "window_name": "YuNet + SFace Fast Many Faces",
        "camera_width": 960,
        "camera_height": 540,
        "yunet_score": 0.70,
        "yunet_nms": 0.30,
        "yunet_top_k": 1000,
        "sface_similarity": 0.36,
        "search_input_width": 640,
        "fast_every": 1,
        "grid_every": 60,
        "enable_grid": False,
        "fast_zooms": [1.0],
        "full_zooms": [1.0, 1.8],
        "candidate_iou": 0.30,
        "draw_unknown": True,
        "suppress_unknown": True,
        "short_labels": True,
        "entrance_mode": False,
        "draw_entrance_zone": False,
    },
    # Based on main_yunet_sface_detect_many_faces_with_unknown.py.
    "many_faces_unknown": {
        "label": "Many faces + unknown review",
        "window_name": "YuNet + SFace Many Faces + Unknown",
        "camera_width": 1280,
        "camera_height": 720,
        "yunet_score": 0.68,
        "yunet_nms": 0.30,
        "yunet_top_k": 5000,
        "sface_similarity": 0.36,
        "search_input_width": 960,
        "fast_every": 2,
        "grid_every": 16,
        "enable_grid": True,
        "fast_zooms": [1.0],
        "full_zooms": [1.0, 1.8, 2.6],
        "candidate_iou": 0.30,
        "draw_unknown": True,
        "suppress_unknown": True,
        "short_labels": False,
        "entrance_mode": False,
        "draw_entrance_zone": False,
    },
    # Based on main_yunet_sface_detect_many_faces_fast_clean.py.
    "fast_clean": {
        "label": "Fast clean known-only display",
        "window_name": "YuNet + SFace Many Faces - Fast Clean",
        "camera_width": 1280,
        "camera_height": 720,
        "yunet_score": 0.68,
        "yunet_nms": 0.30,
        "yunet_top_k": 5000,
        "sface_similarity": 0.36,
        "search_input_width": 960,
        "fast_every": 2,
        "grid_every": 16,
        "enable_grid": True,
        "fast_zooms": [1.0],
        "full_zooms": [1.0, 1.8, 2.6],
        "candidate_iou": 0.30,
        "draw_unknown": False,
        "suppress_unknown": True,
        "short_labels": False,
        "entrance_mode": False,
        "draw_entrance_zone": False,
    },
    # Based on main_yunet_sface_detect_many_faces.py.
    "high_recall_many_faces": {
        "label": "High recall many faces",
        "window_name": "YuNet + SFace Detect Many Faces",
        "camera_width": 1280,
        "camera_height": 720,
        "yunet_score": 0.60,
        "yunet_nms": 0.25,
        "yunet_top_k": 10000,
        "sface_similarity": 0.38,
        "search_input_width": 1280,
        "fast_every": 1,
        "grid_every": 8,
        "enable_grid": True,
        "fast_zooms": [1.0, 1.8, 2.6],
        "full_zooms": [1.0, 1.6, 2.2, 3.0, 4.0],
        "candidate_iou": 0.55,
        "draw_unknown": True,
        "suppress_unknown": False,
        "short_labels": False,
        "entrance_mode": False,
        "draw_entrance_zone": False,
    },
    # Based on main_yunet_sface_multi_attendance.py.
    "multi_attendance_zoom": {
        "label": "Multi-attendance auto-zoom",
        "window_name": "YuNet + SFace Fast Attendance",
        "camera_width": 1280,
        "camera_height": 720,
        "yunet_score": 0.75,
        "yunet_nms": 0.30,
        "yunet_top_k": 5000,
        "sface_similarity": 0.38,
        "search_input_width": 960,
        "fast_every": 2,
        "grid_every": 12,
        "enable_grid": True,
        "fast_zooms": [1.0, 2.4, 3.4],
        "full_zooms": [1.0, 1.8, 2.6, 3.4],
        "candidate_iou": 0.40,
        "draw_unknown": True,
        "suppress_unknown": False,
        "short_labels": False,
        "entrance_mode": False,
        "draw_entrance_zone": False,
    },
    # Based on main_yunet_sface_entrance_mode.py.
    "entrance_mode": {
        "label": "Entrance/door mode",
        "window_name": "YuNet + SFace Entrance Mode",
        "camera_width": 1280,
        "camera_height": 720,
        "yunet_score": 0.75,
        "yunet_nms": 0.30,
        "yunet_top_k": 5000,
        "sface_similarity": 0.38,
        "search_input_width": 960,
        "fast_every": 2,
        "grid_every": 12,
        "enable_grid": True,
        "fast_zooms": [1.0, 2.4, 3.4],
        "full_zooms": [1.0, 1.8, 2.6, 3.4],
        "candidate_iou": 0.40,
        "draw_unknown": True,
        "suppress_unknown": False,
        "short_labels": False,
        "entrance_mode": True,
        "draw_entrance_zone": True,
    },
}


def apply_algorithm_profile(profile_key, source="local"):
    """Apply one of the selectable runtime profiles to the exact OpenCV station.

    The selected profile changes thresholds, search cadence, zoom levels, unknown drawing,
    and entrance-mode behavior. It does not change the actual recognition model family:
    all profiles still use OpenCV YuNet + SFace.
    """
    global ALGORITHM_PROFILE, ACTIVE_ALGORITHM_PROFILE
    global WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT
    global YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K
    global SFACE_SIMILARITY_THRESHOLD, SEARCH_INPUT_WIDTH
    global FAST_SEARCH_EVERY_N_FRAMES, FULL_GRID_SEARCH_EVERY_N_FRAMES, ENABLE_PERIODIC_GRID_SEARCH
    global SEARCH_ZOOM_LEVELS_FAST, SEARCH_ZOOM_LEVELS_FULL
    global CANDIDATE_IOU_DEDUPE_THRESHOLD, DRAW_UNKNOWN_FACES, SUPPRESS_UNKNOWN_NEAR_KNOWN
    global SHORT_LABELS, ENTRANCE_MODE, DRAW_ENTRANCE_ZONE

    key = str(profile_key or "fast_short").strip()
    if key not in ALGORITHM_PROFILES:
        print(f"[WARN] Unknown algorithm profile '{key}', falling back to fast_short")
        key = "fast_short"

    profile = ALGORITHM_PROFILES[key]
    if ACTIVE_ALGORITHM_PROFILE == key:
        return key

    ALGORITHM_PROFILE = key
    ACTIVE_ALGORITHM_PROFILE = key

    WINDOW_NAME = profile.get("window_name", WINDOW_NAME)
    CAMERA_WIDTH = int(profile.get("camera_width", CAMERA_WIDTH))
    CAMERA_HEIGHT = int(profile.get("camera_height", CAMERA_HEIGHT))
    YUNET_SCORE_THRESHOLD = float(profile.get("yunet_score", YUNET_SCORE_THRESHOLD))
    YUNET_NMS_THRESHOLD = float(profile.get("yunet_nms", YUNET_NMS_THRESHOLD))
    YUNET_TOP_K = int(profile.get("yunet_top_k", YUNET_TOP_K))
    SFACE_SIMILARITY_THRESHOLD = float(profile.get("sface_similarity", SFACE_SIMILARITY_THRESHOLD))
    SEARCH_INPUT_WIDTH = int(profile.get("search_input_width", SEARCH_INPUT_WIDTH))
    FAST_SEARCH_EVERY_N_FRAMES = int(profile.get("fast_every", FAST_SEARCH_EVERY_N_FRAMES))
    FULL_GRID_SEARCH_EVERY_N_FRAMES = int(profile.get("grid_every", FULL_GRID_SEARCH_EVERY_N_FRAMES))
    ENABLE_PERIODIC_GRID_SEARCH = bool(profile.get("enable_grid", ENABLE_PERIODIC_GRID_SEARCH))
    SEARCH_ZOOM_LEVELS_FAST = list(profile.get("fast_zooms", SEARCH_ZOOM_LEVELS_FAST))
    SEARCH_ZOOM_LEVELS_FULL = list(profile.get("full_zooms", SEARCH_ZOOM_LEVELS_FULL))
    CANDIDATE_IOU_DEDUPE_THRESHOLD = float(profile.get("candidate_iou", CANDIDATE_IOU_DEDUPE_THRESHOLD))
    DRAW_UNKNOWN_FACES = bool(profile.get("draw_unknown", DRAW_UNKNOWN_FACES))
    SUPPRESS_UNKNOWN_NEAR_KNOWN = bool(profile.get("suppress_unknown", SUPPRESS_UNKNOWN_NEAR_KNOWN))
    SHORT_LABELS = bool(profile.get("short_labels", SHORT_LABELS))
    ENTRANCE_MODE = bool(profile.get("entrance_mode", ENTRANCE_MODE))
    DRAW_ENTRANCE_ZONE = bool(profile.get("draw_entrance_zone", DRAW_ENTRANCE_ZONE))

    print(f"[PROFILE] Using {key}: {profile.get('label', key)} ({source})")
    return key

# Kept only if you still want the old image-loading function for debugging.
ENCODINGS_DIR = "images/"
CAPTURE_DIR = "captures"
WINDOW_NAME = "YuNet + SFace Fast Many Faces"

CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

UNKNOWN_LABEL = "Unknown"

# Models
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


os.makedirs(CAPTURE_DIR, exist_ok=True)


# ============================================================
# Model setup
# ============================================================

def ensure_model_file(path: Path, url: str):
    MODELS_DIR.mkdir(exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        return

    # The Moodle plugin already contains the SFace model for the browser recorder.
    # Reuse it if this script is inside mod/faceattendance/tools/opencv_station.
    plugin_model = SCRIPT_DIR.parent.parent / "uploader" / "models" / path.name
    if plugin_model.exists() and plugin_model.stat().st_size > 0:
        import shutil
        shutil.copyfile(plugin_model, path)
        print(f"Copied model from Moodle plugin: {plugin_model}")
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


def display_name_for_person(person_key):
    """Resolve a recognition key to a human-readable label.

    In Moodle mode the recognition key is the Moodle userid as a string.
    In local JSON mode the key is already the person name.
    """
    key = str(person_key)
    return MOODLE_PERSON_LABELS.get(key, key)


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
                "descriptor": feature.reshape(-1).astype(float).tolist(),
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
# Manual station zoom helpers
# ============================================================

def clamp_float(value, minimum, maximum):
    return max(minimum, min(maximum, float(value)))


def normalize_manual_zoom_state():
    """Keep manual zoom parameters in a safe range."""
    global MANUAL_ZOOM_ENABLED, MANUAL_ZOOM_FACTOR, MANUAL_ZOOM_CENTER_X, MANUAL_ZOOM_CENTER_Y

    MANUAL_ZOOM_FACTOR = clamp_float(MANUAL_ZOOM_FACTOR, MANUAL_ZOOM_MIN, MANUAL_ZOOM_MAX)
    if MANUAL_ZOOM_FACTOR <= 1.001:
        MANUAL_ZOOM_FACTOR = 1.0
        MANUAL_ZOOM_ENABLED = False

    MANUAL_ZOOM_CENTER_X = clamp_float(MANUAL_ZOOM_CENTER_X, 0.0, 1.0)
    MANUAL_ZOOM_CENTER_Y = clamp_float(MANUAL_ZOOM_CENTER_Y, 0.0, 1.0)


def manual_zoom_crop_params(frame_w, frame_h):
    """Return x, y, width, height for the current manual zoom crop."""
    normalize_manual_zoom_state()

    if not MANUAL_ZOOM_ENABLED or MANUAL_ZOOM_FACTOR <= 1.001:
        return 0, 0, frame_w, frame_h

    crop_w = max(1, int(round(frame_w / MANUAL_ZOOM_FACTOR)))
    crop_h = max(1, int(round(frame_h / MANUAL_ZOOM_FACTOR)))

    cx = MANUAL_ZOOM_CENTER_X * frame_w
    cy = MANUAL_ZOOM_CENTER_Y * frame_h

    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))

    left = max(0, min(frame_w - crop_w, left))
    top = max(0, min(frame_h - crop_h, top))

    return left, top, crop_w, crop_h


def apply_manual_station_zoom(frame):
    """Apply digital zoom before detection and display.

    The output has the same size as the camera frame, but it contains only the
    selected crop enlarged back to full resolution. Detection, recognition,
    thumbnails, and proof images then operate on the zoomed view.
    """
    frame_h, frame_w = frame.shape[:2]
    left, top, crop_w, crop_h = manual_zoom_crop_params(frame_w, frame_h)

    if crop_w == frame_w and crop_h == frame_h:
        return frame, (left, top, crop_w, crop_h)

    crop = frame[top:top + crop_h, left:left + crop_w]
    zoomed = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
    return zoomed, (left, top, crop_w, crop_h)


def handle_manual_zoom_key(key):
    """Handle keyboard controls for manual digital zoom.

    Returns True when the key changed zoom state.
    """
    global MANUAL_ZOOM_ENABLED, MANUAL_ZOOM_FACTOR, MANUAL_ZOOM_CENTER_X, MANUAL_ZOOM_CENTER_Y

    if key < 0:
        return False

    changed = False

    if key in (ord("m"), ord("M")):
        MANUAL_ZOOM_ENABLED = not MANUAL_ZOOM_ENABLED
        if MANUAL_ZOOM_ENABLED and MANUAL_ZOOM_FACTOR <= 1.001:
            MANUAL_ZOOM_FACTOR = 1.5
        changed = True

    elif key in (ord("+"), ord("=")):
        MANUAL_ZOOM_ENABLED = True
        MANUAL_ZOOM_FACTOR = min(MANUAL_ZOOM_MAX, MANUAL_ZOOM_FACTOR + MANUAL_ZOOM_STEP)
        changed = True

    elif key in (ord("-"), ord("_")):
        MANUAL_ZOOM_FACTOR = max(MANUAL_ZOOM_MIN, MANUAL_ZOOM_FACTOR - MANUAL_ZOOM_STEP)
        if MANUAL_ZOOM_FACTOR <= 1.001:
            MANUAL_ZOOM_FACTOR = 1.0
            MANUAL_ZOOM_ENABLED = False
        changed = True

    elif key == ord("0"):
        MANUAL_ZOOM_ENABLED = False
        MANUAL_ZOOM_FACTOR = 1.0
        MANUAL_ZOOM_CENTER_X = 0.5
        MANUAL_ZOOM_CENTER_Y = 0.5
        changed = True

    elif key in (ord("w"), ord("W")) and MANUAL_ZOOM_ENABLED:
        MANUAL_ZOOM_CENTER_Y -= MANUAL_ZOOM_PAN_STEP / max(1.0, MANUAL_ZOOM_FACTOR)
        changed = True

    elif key in (ord("s"), ord("S")) and MANUAL_ZOOM_ENABLED:
        MANUAL_ZOOM_CENTER_Y += MANUAL_ZOOM_PAN_STEP / max(1.0, MANUAL_ZOOM_FACTOR)
        changed = True

    elif key in (ord("a"), ord("A")) and MANUAL_ZOOM_ENABLED:
        MANUAL_ZOOM_CENTER_X -= MANUAL_ZOOM_PAN_STEP / max(1.0, MANUAL_ZOOM_FACTOR)
        changed = True

    elif key in (ord("d"), ord("D")) and MANUAL_ZOOM_ENABLED:
        MANUAL_ZOOM_CENTER_X += MANUAL_ZOOM_PAN_STEP / max(1.0, MANUAL_ZOOM_FACTOR)
        changed = True

    if changed:
        normalize_manual_zoom_state()
        state = "ON" if MANUAL_ZOOM_ENABLED else "OFF"
        print(
            f"Manual zoom: {state} | {MANUAL_ZOOM_FACTOR:.2f}x "
            f"| center=({MANUAL_ZOOM_CENTER_X:.2f}, {MANUAL_ZOOM_CENTER_Y:.2f})"
        )

    return changed


def manual_zoom_status_text():
    if not MANUAL_ZOOM_ENABLED or MANUAL_ZOOM_FACTOR <= 1.001:
        return "zoom OFF"
    return f"zoom {MANUAL_ZOOM_FACTOR:.2f}x @ {MANUAL_ZOOM_CENTER_X:.2f},{MANUAL_ZOOM_CENTER_Y:.2f}"

# ============================================================
# Runtime configuration helpers
# ============================================================

def _load_json_config(path):
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise RuntimeError(f"Config file does not exist: {config_path.resolve()}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError("Station config file must contain a JSON object.")
    return data




def bool_config_value(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in ("0", "false", "no", "off", "")

def apply_runtime_options():
    """Override constants from command-line arguments or station_config.json.

    Priority:
      1. command-line argument
      2. station_config.json value
      3. environment variable/default from the constants above
    """
    global MOODLE_BASE_URL, MOODLE_CMID, MOODLE_API_SECRET, MOODLE_STATION_SOURCE
    global CAMERA_INDEX, MOODLE_ALGORITHM_PROFILE, ALGORITHM_PROFILE
    global MOODLE_REFRESH_SECONDS, UNKNOWN_MOODLE_COOLDOWN_SECONDS, MOODLE_HTTP_TIMEOUT_SECONDS, MOODLE_VERIFY_TLS
    global USE_MOODLE_API
    global MANUAL_ZOOM_ENABLED, MANUAL_ZOOM_FACTOR, MANUAL_ZOOM_MAX, MANUAL_ZOOM_CENTER_X, MANUAL_ZOOM_CENTER_Y, MANUAL_ZOOM_STEP, MANUAL_ZOOM_PAN_STEP

    parser = argparse.ArgumentParser(description="Moodle Face Attendance OpenCV YuNet/SFace camera station")
    parser.add_argument("--config", default=os.environ.get("FACEATTENDANCE_CONFIG"), help="Path to station_config.json")
    parser.add_argument("--moodle-url", default=None, help="Moodle base URL, e.g. https://192.168.0.154")
    parser.add_argument("--cmid", type=int, default=None, help="Moodle Face Attendance course module id")
    parser.add_argument("--secret", default=None, help="Face Attendance API secret from the activity settings")
    parser.add_argument("--camera", type=int, default=None, help="OpenCV camera index, usually 0 or 1")
    parser.add_argument("--profile", default=None, help="Algorithm profile: fast_short, high_recall_many_faces, entrance_mode, etc.")
    parser.add_argument("--zoom", type=float, default=None, help="Initial manual digital zoom factor. Example: --zoom 2.0")
    parser.add_argument("--zoom-center-x", type=float, default=None, help="Initial zoom center X as 0.0-1.0. Example: 0.5")
    parser.add_argument("--zoom-center-y", type=float, default=None, help="Initial zoom center Y as 0.0-1.0. Example: 0.5")
    parser.add_argument("--source", default=None, help="Station source name shown in Moodle logs")
    parser.add_argument("--no-moodle", action="store_true", help="Run without Moodle API, using local EMBEDDINGS_SOURCE")
    args = parser.parse_args()

    config = {}

    def resolve_config_path(config_arg):
        candidates = []
        if config_arg:
            raw = Path(config_arg)
            if raw.is_absolute():
                candidates.append(raw)
            else:
                candidates.append(Path.cwd() / raw)
                candidates.append(SCRIPT_DIR / raw)
        else:
            candidates.append(Path.cwd() / "station_config.json")
            candidates.append(SCRIPT_DIR / "station_config.json")

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        if config_arg:
            checked = "\n".join(f"  - {c.resolve()}" for c in candidates)
            raise RuntimeError("Config file was explicitly requested but not found. Checked:\n" + checked)
        return None

    config_path = resolve_config_path(args.config)
    if config_path is not None:
        print(f"[CONFIG] Loaded station config: {config_path}")
        config = _load_json_config(config_path)
    else:
        print("[CONFIG] No station_config.json found. Using built-in defaults and command-line/env values.")

    def pick(key, argvalue, current):
        return argvalue if argvalue is not None else config.get(key, current)

    MOODLE_BASE_URL = str(pick("moodle_base_url", args.moodle_url, MOODLE_BASE_URL)).rstrip("/")
    MOODLE_CMID = int(pick("cmid", args.cmid, MOODLE_CMID))
    MOODLE_API_SECRET = str(pick("api_secret", args.secret, MOODLE_API_SECRET))
    CAMERA_INDEX = int(pick("camera_index", args.camera, CAMERA_INDEX))
    MOODLE_STATION_SOURCE = str(pick("source", args.source, MOODLE_STATION_SOURCE))
    MOODLE_REFRESH_SECONDS = int(float(config.get("refresh_seconds", MOODLE_REFRESH_SECONDS)))
    UNKNOWN_MOODLE_COOLDOWN_SECONDS = int(float(config.get("unknown_cooldown_seconds", UNKNOWN_MOODLE_COOLDOWN_SECONDS)))
    MOODLE_HTTP_TIMEOUT_SECONDS = int(float(config.get("http_timeout_seconds", MOODLE_HTTP_TIMEOUT_SECONDS)))
    MOODLE_VERIFY_TLS = bool_config_value(config.get("verify_tls", MOODLE_VERIFY_TLS), MOODLE_VERIFY_TLS)

    MANUAL_ZOOM_ENABLED = bool_config_value(config.get("manual_zoom_enabled", MANUAL_ZOOM_ENABLED), MANUAL_ZOOM_ENABLED)
    MANUAL_ZOOM_FACTOR = float(pick("manual_zoom", args.zoom, MANUAL_ZOOM_FACTOR))
    MANUAL_ZOOM_MAX = float(config.get("manual_zoom_max", MANUAL_ZOOM_MAX))
    MANUAL_ZOOM_CENTER_X = float(pick("manual_zoom_center_x", args.zoom_center_x, MANUAL_ZOOM_CENTER_X))
    MANUAL_ZOOM_CENTER_Y = float(pick("manual_zoom_center_y", args.zoom_center_y, MANUAL_ZOOM_CENTER_Y))
    MANUAL_ZOOM_STEP = float(config.get("manual_zoom_step", MANUAL_ZOOM_STEP))
    MANUAL_ZOOM_PAN_STEP = float(config.get("manual_zoom_pan_step", MANUAL_ZOOM_PAN_STEP))
    if args.zoom is not None and MANUAL_ZOOM_FACTOR > 1.001:
        MANUAL_ZOOM_ENABLED = True
    normalize_manual_zoom_state()

    profile = pick("profile", args.profile, MOODLE_ALGORITHM_PROFILE)
    if profile not in ALGORITHM_PROFILES:
        print(f"[WARN] Unknown profile '{profile}', using fast_short.")
        profile = "fast_short"
    MOODLE_ALGORITHM_PROFILE = profile
    ALGORITHM_PROFILE = profile

    if args.no_moodle or bool(config.get("no_moodle", False)):
        USE_MOODLE_API = False

    print("\nStation configuration:")
    print(f"  Moodle API enabled: {USE_MOODLE_API}")
    print(f"  Moodle URL: {MOODLE_BASE_URL}")
    print(f"  CMID: {MOODLE_CMID}")
    print(f"  Camera index: {CAMERA_INDEX}")
    print(f"  Profile: {MOODLE_ALGORITHM_PROFILE}")
    print(f"  Source: {MOODLE_STATION_SOURCE}")
    print(f"  Verify TLS: {MOODLE_VERIFY_TLS}")
    print(f"  Manual zoom: {manual_zoom_status_text()}\n")

# ============================================================
# Moodle API helpers for external Python camera station
# ============================================================

def moodle_endpoint(path):
    base = MOODLE_BASE_URL.rstrip("/")
    return f"{base}/mod/faceattendance/api/{path}"


def moodle_ssl_context():
    if MOODLE_VERIFY_TLS:
        return None
    return ssl._create_unverified_context()


def http_json_get(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=MOODLE_HTTP_TIMEOUT_SECONDS, context=moodle_ssl_context()) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from Moodle: {body}") from exc


def http_json_post(url, payload, headers=None):
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=MOODLE_HTTP_TIMEOUT_SECONDS, context=moodle_ssl_context()) as response:
            data = response.read().decode("utf-8")
            return json.loads(data)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from Moodle: {body}") from exc


def fetch_moodle_bootstrap():
    query = urllib.parse.urlencode({
        "cmid": MOODLE_CMID,
        "secret": MOODLE_API_SECRET,
        "profile": MOODLE_ALGORITHM_PROFILE,
    })
    url = moodle_endpoint(f"station_bootstrap.php?{query}")
    return http_json_get(url, headers={"Accept": "application/json"})


def load_known_features_from_moodle():
    """Load Moodle-registered SFace descriptors into the same structure used by the original script."""
    global MOODLE_PERSON_LABELS, MOODLE_ACTIVE_SESSION, MOODLE_LAST_BOOTSTRAP

    payload = fetch_moodle_bootstrap()
    if not payload.get("ok"):
        raise RuntimeError(f"Moodle bootstrap failed: {payload}")

    payload_profile = payload.get("modelProfile") or payload.get("modelprofile")
    if payload_profile:
        apply_algorithm_profile(payload_profile, source="moodle")

    MOODLE_LAST_BOOTSTRAP = time.time()
    MOODLE_ACTIVE_SESSION = payload.get("session") if payload.get("active") else None
    faces = payload.get("faces") or []

    known_features: Dict[str, List[np.ndarray]] = {}
    labels = {}

    for face in faces:
        userid = face.get("userid") or face.get("studentId")
        descriptor = face.get("descriptor")
        if userid is None or not is_embedding_vector(descriptor):
            continue

        person_key = str(userid)
        labels[person_key] = str(face.get("name") or person_key)
        add_embedding(known_features, person_key, descriptor, f"Moodle userid {person_key}")

    MOODLE_PERSON_LABELS = labels

    print(f"Loaded {sum(len(v) for v in known_features.values())} embedding sample(s) from Moodle.")
    print(f"Loaded {len(known_features)} Moodle user(s).")
    if MOODLE_ACTIVE_SESSION:
        print(f"Active Moodle session: {MOODLE_ACTIVE_SESSION.get('name')} (id={MOODLE_ACTIVE_SESSION.get('id')})")
    else:
        print("No active Moodle attendance session right now. The camera will keep running and refresh periodically.")

    if not known_features:
        raise RuntimeError("Moodle returned no registered embeddings. Register at least one student first.")

    return known_features


def refresh_moodle_state_if_needed(current_known_features):
    global MOODLE_LAST_BOOTSTRAP

    if not USE_MOODLE_API:
        return current_known_features

    now = time.time()
    if now - MOODLE_LAST_BOOTSTRAP < MOODLE_REFRESH_SECONDS:
        return current_known_features

    try:
        return load_known_features_from_moodle()
    except Exception as exc:
        print(f"[WARN] Could not refresh Moodle state: {exc}")
        MOODLE_LAST_BOOTSTRAP = now
        return current_known_features


def current_moodle_session_id():
    if not MOODLE_ACTIVE_SESSION:
        return None
    return int(MOODLE_ACTIVE_SESSION.get("id") or 0) or None


def post_moodle_attendance(person_key, similarity, distance, photo_path):
    sessionid = current_moodle_session_id()
    if not sessionid:
        print("[MOODLE] Known face detected, but there is no active session. Not marking attendance.")
        return None

    userid = int(person_key)
    payload = {
        "cmid": MOODLE_CMID,
        "secret": MOODLE_API_SECRET,
        "sessionid": sessionid,
        "userid": userid,
        "confidence": float(similarity or 0.0),
        "distance": float(distance or 0.0),
        "source": MOODLE_STATION_SOURCE,
        "photo_path": photo_path,
    }
    result = http_json_post(moodle_endpoint("station_mark.php"), payload)
    if not result.get("ok"):
        raise RuntimeError(f"Moodle mark failed: {result}")
    return result



def make_unknown_thumbnail_data_url(frame, box):
    """Returns a small JPEG data URL for teacher review, or None if crop fails."""
    if frame is None or box is None:
        return None

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad = int(max(box_w, box_h) * UNKNOWN_THUMBNAIL_PADDING_RATIO)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame_w, x2 + pad)
    y2 = min(frame_h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_h, crop_w = crop.shape[:2]
    if crop_w > UNKNOWN_THUMBNAIL_MAX_WIDTH:
        scale = UNKNOWN_THUMBNAIL_MAX_WIDTH / float(crop_w)
        crop = cv2.resize(
            crop,
            (UNKNOWN_THUMBNAIL_MAX_WIDTH, max(1, int(round(crop_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(UNKNOWN_THUMBNAIL_JPEG_QUALITY)],
    )
    if not ok:
        return None

    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + b64

def post_moodle_unknown(candidate, frame=None):
    sessionid = current_moodle_session_id()
    if not sessionid:
        return None

    descriptor = candidate.get("descriptor")
    if not is_embedding_vector(descriptor):
        return None

    x1, y1, x2, y2 = [int(v) for v in candidate.get("box", (0, 0, 0, 0))]
    payload = {
        "cmid": MOODLE_CMID,
        "secret": MOODLE_API_SECRET,
        "sessionid": sessionid,
        "descriptor": [float(v) for v in descriptor],
        "source": MOODLE_STATION_SOURCE,
        "candidate": {
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "similarity": float(candidate.get("similarity", 0.0)),
            "source": candidate.get("source", "opencv"),
        },
    }

    thumbnail = make_unknown_thumbnail_data_url(frame, candidate.get("box"))
    if thumbnail:
        payload["thumbnail"] = thumbnail
    result = http_json_post(moodle_endpoint("station_unknown.php"), payload)
    if not result.get("ok"):
        raise RuntimeError(f"Moodle unknown save failed: {result}")
    return result

# ============================================================
# Attendance/photo helpers
# ============================================================

def save_person_photo(name, zoomed_frame):
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    image_path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}_person.jpg")
    cv2.imwrite(image_path, zoomed_frame)

    return image_path


def mark_attendance(name, photo_path, similarity=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    display_name = display_name_for_person(name)

    if USE_MOODLE_API:
        try:
            similarity_value = float(similarity or 0.0)
            # SFace uses cosine similarity. A simple diagnostic distance is 1 - similarity.
            distance_value = max(0.0, 1.0 - similarity_value)
            result = post_moodle_attendance(name, similarity_value, distance_value, photo_path)
            if result:
                print(f"[MOODLE] {result.get('saved')} attendance for {result.get('name')} as {result.get('status')}")
        except Exception as exc:
            print(f"[ERROR] Could not send attendance to Moodle: {exc}")

    # Keep the local CSV as a backup/debug audit file.
    file_exists = os.path.exists(ATTENDANCE_CSV)
    with open(ATTENDANCE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["timestamp", "name", "moodle_userid", "similarity", "photo_path"])

        writer.writerow([timestamp, display_name, name, similarity if similarity is not None else "", photo_path])

    return timestamp


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
    apply_algorithm_profile(ALGORITHM_PROFILE, source="startup")
    ensure_model_file(YUNET_MODEL, YUNET_URL)
    ensure_model_file(SFACE_MODEL, SFACE_URL)

    detector = create_detector()
    recognizer = create_recognizer()

    if USE_MOODLE_API:
        known_features = load_known_features_from_moodle()
    else:
        known_features = load_known_features_from_embeddings(EMBEDDINGS_SOURCE)

    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try another camera index, for example --camera 1.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"Algorithm profile: {ALGORITHM_PROFILE}")
    print(f"YuNet score threshold: {YUNET_SCORE_THRESHOLD}")
    print(f"SFace similarity threshold: {SFACE_SIMILARITY_THRESHOLD}")
    print("Press ESC to exit. Press R to reset tracking state.")
    print("Fast mode: full-frame detection, unknown faces enabled, periodic grid search OFF by default.")
    print("Press U to toggle Unknown drawing. Press G to toggle slower grid/zoom search.")
    print("Manual zoom controls: M toggle | + zoom in | - zoom out | 0 reset | W/A/S/D pan.\n")

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
    last_unknown_moodle_save_time = 0.0

    frame_index = 0
    last_candidates = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        # Optional digital zoom is applied before detection/recognition.
        # This is useful when the camera is aimed at the classroom entrance
        # and faces are too small in the full frame.
        frame, manual_zoom_crop = apply_manual_station_zoom(frame)

        now = time.time()
        frame_index += 1
        frame_h, frame_w = frame.shape[:2]

        known_features = refresh_moodle_state_if_needed(known_features)

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

        # Full-frame display: do not zoom into one face, because that hides other people.
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
                    image_path = save_person_photo(display_name_for_person(name), clean_zoomed)
                    timestamp = mark_attendance(name, image_path, candidate.get("similarity"))

                    last_capture_time_by_name[name] = now
                    last_attendance_time_by_name[name] = now
                    saved_count_by_name[name] += 1

                    print(f"ATTENDANCE MARKED: {display_name_for_person(name)} at {timestamp}")
                    print(f"Saved proof photo: {image_path}")

        # Send unknown faces to Moodle for teacher review, but throttle to avoid flooding the server.
        if USE_MOODLE_API and current_moodle_session_id() and (now - last_unknown_moodle_save_time >= UNKNOWN_MOODLE_COOLDOWN_SECONDS):
            for candidate in candidates:
                if candidate.get("is_known", False):
                    continue
                try:
                    result = post_moodle_unknown(candidate, frame)
                    if result:
                        last_unknown_moodle_save_time = now
                        print(f"[MOODLE] Saved unknown face #{result.get('unknownid')} for teacher review")
                        break
                except Exception as exc:
                    last_unknown_moodle_save_time = now
                    print(f"[WARN] Could not save unknown face to Moodle: {exc}")
                    break

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
                label = short_display_name(display_name_for_person(name))
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

        status += f" | profile {ALGORITHM_PROFILE} | {manual_zoom_status_text()}"

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
            "ESC exit | R reset | U unknown | G grid | M zoom | +/- zoom | WASD pan",
            (20, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if handle_manual_zoom_key(key):
            last_candidates = []

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    apply_runtime_options()
    main()
