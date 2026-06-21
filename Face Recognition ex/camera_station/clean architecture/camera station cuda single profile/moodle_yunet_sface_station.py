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
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

# Self-bootstrap runtime packages and NVIDIA CUDA DLL search paths before
# importing cv2/ONNX Runtime.  This makes direct `py <script>.py` launches
# repair the Python environment automatically.
try:
    from faceattendance_runtime_bootstrap import ensure_faceattendance_runtime
    ensure_faceattendance_runtime(include_gui=False, prefer_gpu=True, verbose=True)
except Exception as _bootstrap_exc:
    print(f"[BOOTSTRAP] Runtime bootstrap warning: {_bootstrap_exc}")

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
    # Optimized for 20-40 people in the same frame. It avoids grid/zoom search,
    # searches at a smaller input size, skips already resolved faces, and spreads
    # SFace recognition over frames if too many unresolved faces remain.
    "crowd_fast": {
        "label": "Crowd fast / 20-40 faces",
        "window_name": "YuNet + SFace Crowd Fast",
        "camera_width": 960,
        "camera_height": 540,
        "yunet_score": 0.72,
        "yunet_nms": 0.35,
        "yunet_top_k": 1000,
        "sface_similarity": 0.36,
        "search_input_width": 512,
        "fast_every": 2,
        "grid_every": 120,
        "enable_grid": False,
        "fast_zooms": [1.0],
        "full_zooms": [1.0],
        "candidate_iou": 0.35,
        "draw_unknown": True,
        "suppress_unknown": True,
        "short_labels": True,
        "entrance_mode": False,
        "draw_entrance_zone": False,
        "skip_resolved_recognition": True,
        "max_recognitions_per_frame": 18,
    },
    # Emergency performance profile for large classrooms or weak CPU.
    # It intentionally lowers camera/search resolution and spreads recognition
    # across frames. Use this when 25-40+ faces make the UI lag.
    "crowd_turbo": {
        "label": "Crowd turbo / emergency speed",
        "window_name": "YuNet + SFace Crowd Turbo",
        "camera_width": 640,
        "camera_height": 360,
        "yunet_score": 0.76,
        "yunet_nms": 0.36,
        "yunet_top_k": 300,
        "sface_similarity": 0.36,
        "search_input_width": 384,
        "fast_every": 3,
        "grid_every": 9999,
        "enable_grid": False,
        "fast_zooms": [1.0],
        "full_zooms": [1.0],
        "candidate_iou": 0.35,
        "draw_unknown": True,
        "suppress_unknown": True,
        "short_labels": True,
        "entrance_mode": False,
        "draw_entrance_zone": False,
        "skip_resolved_recognition": True,
        "max_recognitions_per_frame": 8,
    },
    # Extreme classroom speed profile. This is for diagnosing/handling scenes
    # with 30-40+ faces where the CPU detector/UI, not SFace, is the bottleneck.
    "crowd_extreme": {
        "label": "Crowd extreme / CPU detector relief",
        "window_name": "YuNet + SFace Crowd Extreme",
        "camera_width": 640,
        "camera_height": 360,
        "yunet_score": 0.82,
        "yunet_nms": 0.40,
        "yunet_top_k": 120,
        "sface_similarity": 0.36,
        "search_input_width": 320,
        "fast_every": 5,
        "grid_every": 9999,
        "enable_grid": False,
        "fast_zooms": [1.0],
        "full_zooms": [1.0],
        "candidate_iou": 0.40,
        "draw_unknown": False,
        "suppress_unknown": True,
        "short_labels": True,
        "entrance_mode": False,
        "draw_entrance_zone": False,
        "skip_resolved_recognition": True,
        "max_recognitions_per_frame": 24,
        "unknown_registry_max_updates_per_frame": 0,
        "desktop_output_every_n_frames": 2,
        "desktop_report_every_seconds": 0.75,
    },
    # Real-time walk-through profile. Use this for an entrance/corridor camera
    # where people are moving normally and should be labelled without stopping.
    # It keeps detection on every frame and lowers the stability requirement,
    # while still using SFace similarity for identity verification.
    "walkthrough_realtime": {
        "label": "Walk-through realtime / moving people",
        "window_name": "YuNet + SFace Walk-through Realtime",
        "camera_width": 960,
        "camera_height": 540,
        "yunet_score": 0.66,
        "yunet_nms": 0.34,
        "yunet_top_k": 500,
        "sface_similarity": 0.36,
        "search_input_width": 640,
        "fast_every": 1,
        "grid_every": 9999,
        "enable_grid": False,
        "fast_zooms": [1.0],
        "full_zooms": [1.0],
        "candidate_iou": 0.32,
        "draw_unknown": False,
        "suppress_unknown": True,
        "short_labels": True,
        "entrance_mode": False,
        "draw_entrance_zone": False,
        "skip_resolved_recognition": True,
        "max_recognitions_per_frame": 24,
        "unknown_registry_max_updates_per_frame": 0,
        "desktop_output_every_n_frames": 1,
        "desktop_report_every_seconds": 0.5,
        "stable_frames_required": 2,
        "confirmed_known_stable_frames": 2,
        "confirmed_known_similarity_threshold": 0.50,
        "confirmed_unknown_stable_frames": 5,
        "face_state_timeout_seconds": 1.2,
        "photo_cooldown_seconds": 2.0,
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
    global SKIP_RESOLVED_FACE_RECOGNITION, MAX_RECOGNITIONS_PER_FRAME
    global UNKNOWN_REGISTRY_MAX_UPDATES_PER_FRAME, DESKTOP_OUTPUT_EVERY_N_FRAMES, DESKTOP_REPORT_EVERY_SECONDS
    global STABLE_FRAMES_REQUIRED, CONFIRMED_KNOWN_STABLE_FRAMES, CONFIRMED_KNOWN_SIMILARITY_THRESHOLD
    global CONFIRMED_UNKNOWN_STABLE_FRAMES, FACE_STATE_TIMEOUT_SECONDS, PHOTO_COOLDOWN_SECONDS

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
    SKIP_RESOLVED_FACE_RECOGNITION = bool(profile.get("skip_resolved_recognition", SKIP_RESOLVED_FACE_RECOGNITION))
    MAX_RECOGNITIONS_PER_FRAME = int(profile.get("max_recognitions_per_frame", MAX_RECOGNITIONS_PER_FRAME))
    UNKNOWN_REGISTRY_MAX_UPDATES_PER_FRAME = int(profile.get("unknown_registry_max_updates_per_frame", UNKNOWN_REGISTRY_MAX_UPDATES_PER_FRAME))
    DESKTOP_OUTPUT_EVERY_N_FRAMES = max(1, int(profile.get("desktop_output_every_n_frames", DESKTOP_OUTPUT_EVERY_N_FRAMES)))
    DESKTOP_REPORT_EVERY_SECONDS = max(0.10, float(profile.get("desktop_report_every_seconds", DESKTOP_REPORT_EVERY_SECONDS)))

    # Stability controls.  These are deliberately profile-driven because moving
    # people should not have to stop for 1-2 seconds just to accumulate 5 stable
    # frames.  The walk-through profile uses 2 frames for recognition/attendance
    # stability and still keeps a strict SFace similarity check.
    STABLE_FRAMES_REQUIRED = max(1, int(profile.get("stable_frames_required", STABLE_FRAMES_REQUIRED)))
    CONFIRMED_KNOWN_STABLE_FRAMES = max(1, int(profile.get("confirmed_known_stable_frames", CONFIRMED_KNOWN_STABLE_FRAMES)))
    CONFIRMED_KNOWN_SIMILARITY_THRESHOLD = float(profile.get("confirmed_known_similarity_threshold", CONFIRMED_KNOWN_SIMILARITY_THRESHOLD))
    CONFIRMED_UNKNOWN_STABLE_FRAMES = max(1, int(profile.get("confirmed_unknown_stable_frames", CONFIRMED_UNKNOWN_STABLE_FRAMES)))
    FACE_STATE_TIMEOUT_SECONDS = max(0.2, float(profile.get("face_state_timeout_seconds", FACE_STATE_TIMEOUT_SECONDS)))
    PHOTO_COOLDOWN_SECONDS = max(0.0, float(profile.get("photo_cooldown_seconds", PHOTO_COOLDOWN_SECONDS)))

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

# After a known person is repeatedly recognized with high confidence, hide that
# person's box/label from the live view. The detector will still see the face,
# but the UI and review flow stay focused on the remaining unresolved faces.
HIDE_CONFIRMED_KNOWN_FACES = os.environ.get("FACEATTENDANCE_HIDE_CONFIRMED_KNOWN", "true").lower() not in ("0", "false", "no", "off")
CONFIRMED_KNOWN_SIMILARITY_THRESHOLD = float(os.environ.get("FACEATTENDANCE_CONFIRMED_SIMILARITY", "0.52"))
CONFIRMED_KNOWN_STABLE_FRAMES = int(float(os.environ.get("FACEATTENDANCE_CONFIRMED_STABLE_FRAMES", str(STABLE_FRAMES_REQUIRED))))

# After the same Unknown face has been seen enough times, hide it from the live
# view and stop treating it as an unresolved visual target. Unknown tracking is
# descriptor-based, not only box-position-based, so it still works while a person
# moves slightly in front of the camera.
HIDE_CONFIRMED_UNKNOWN_FACES = os.environ.get("FACEATTENDANCE_HIDE_CONFIRMED_UNKNOWN", "true").lower() not in ("0", "false", "no", "off")
CONFIRMED_UNKNOWN_STABLE_FRAMES = int(float(os.environ.get("FACEATTENDANCE_CONFIRMED_UNKNOWN_FRAMES", "10")))
UNKNOWN_TRACK_MATCH_THRESHOLD = float(os.environ.get("FACEATTENDANCE_UNKNOWN_TRACK_THRESHOLD", "0.50"))
UNKNOWN_TRACK_DESCRIPTOR_ALPHA = float(os.environ.get("FACEATTENDANCE_UNKNOWN_TRACK_ALPHA", "0.15"))

# Known-identity voting/hysteresis. SFace can produce occasional one-frame false
# matches when the embedding pool is large. Without a per-face vote gate, those
# transient labels pollute Reports > Known persons even when only one real person
# is in front of the camera. The vote tracker below confirms an identity only
# after the same physical face receives repeated, spatially-consistent votes.
IDENTITY_VOTING_ENABLED = os.environ.get("FACEATTENDANCE_IDENTITY_VOTING", "true").lower() not in ("0", "false", "no", "off")
IDENTITY_TRACK_TIMEOUT_SECONDS = float(os.environ.get("FACEATTENDANCE_IDENTITY_TRACK_TIMEOUT", "2.0"))
IDENTITY_TRACK_IOU_THRESHOLD = float(os.environ.get("FACEATTENDANCE_IDENTITY_TRACK_IOU", "0.08"))
IDENTITY_TRACK_CENTER_RATIO = float(os.environ.get("FACEATTENDANCE_IDENTITY_TRACK_CENTER_RATIO", "0.85"))
IDENTITY_TRACK_DESCRIPTOR_THRESHOLD = float(os.environ.get("FACEATTENDANCE_IDENTITY_TRACK_DESCRIPTOR_THRESHOLD", "0.35"))
IDENTITY_TRACK_DESCRIPTOR_ALPHA = float(os.environ.get("FACEATTENDANCE_IDENTITY_TRACK_ALPHA", "0.25"))
IDENTITY_VOTE_WINDOW = int(float(os.environ.get("FACEATTENDANCE_IDENTITY_VOTE_WINDOW", "12")))
IDENTITY_MIN_KNOWN_VOTES = int(float(os.environ.get("FACEATTENDANCE_IDENTITY_MIN_VOTES", "4")))
IDENTITY_MIN_KNOWN_VOTE_RATIO = float(os.environ.get("FACEATTENDANCE_IDENTITY_MIN_RATIO", "0.60"))
IDENTITY_MIN_KNOWN_MARGIN_VOTES = int(float(os.environ.get("FACEATTENDANCE_IDENTITY_MIN_MARGIN_VOTES", "2")))
IDENTITY_MIN_ACCEPT_SIMILARITY = float(os.environ.get("FACEATTENDANCE_IDENTITY_MIN_SIMILARITY", "0.40"))
IDENTITY_MIN_ACCEPT_MARGIN = float(os.environ.get("FACEATTENDANCE_IDENTITY_MIN_MARGIN", "0.015"))
IDENTITY_SWITCH_MIN_VOTES = int(float(os.environ.get("FACEATTENDANCE_IDENTITY_SWITCH_MIN_VOTES", "7")))
IDENTITY_SWITCH_MIN_RATIO = float(os.environ.get("FACEATTENDANCE_IDENTITY_SWITCH_MIN_RATIO", "0.75"))
IDENTITY_SWITCH_MARGIN_VOTES = int(float(os.environ.get("FACEATTENDANCE_IDENTITY_SWITCH_MARGIN_VOTES", "3")))

# Performance option for crowded rooms. Once a known/unknown face is confirmed
# and hidden, future detections that overlap that face reuse the existing track
# instead of running SFace again. This keeps the UI focused on unresolved faces
# and greatly reduces per-frame recognition cost after the first seconds.
SKIP_RESOLVED_FACE_RECOGNITION = os.environ.get("FACEATTENDANCE_SKIP_RESOLVED_RECOGNITION", "true").lower() not in ("0", "false", "no", "off")
RESOLVED_FACE_SKIP_IOU = float(os.environ.get("FACEATTENDANCE_RESOLVED_SKIP_IOU", "0.18"))
RESOLVED_FACE_SKIP_CENTER_RATIO = float(os.environ.get("FACEATTENDANCE_RESOLVED_SKIP_CENTER_RATIO", "0.70"))

# 0 means recognize every non-resolved detected face. In crowded rooms you can
# set this to a value such as 12-20 to cap SFace work per frame. The default is
# unlimited to preserve the original behavior.
MAX_RECOGNITIONS_PER_FRAME = int(float(os.environ.get("FACEATTENDANCE_MAX_RECOGNITIONS_PER_FRAME", "0")))

# 0 means never stop a known person by detection count. A value such as 150
# makes a person "done" after that many accepted known recognitions. After this
# point the UI stops increasing that person's counter and the detector can reuse
# the last box without running SFace for that face again.
KNOWN_STOP_AFTER_DETECTIONS = int(float(os.environ.get("FACEATTENDANCE_KNOWN_STOP_AFTER_DETECTIONS", "150")))

# Desktop review performance: do not update/write the unknown-review JSON for
# every unknown face in every frame. Cropping + cv2.imwrite + JSON persistence
# can dominate CPU/disk time in a crowded scene.
UNKNOWN_REGISTRY_MAX_UPDATES_PER_FRAME = int(float(os.environ.get("FACEATTENDANCE_UNKNOWN_REGISTRY_MAX_UPDATES_PER_FRAME", "2")))

# Lightweight runtime counters used by the desktop UI to show whether the slow
# part is detection or SFace recognition.
LAST_DETECTED_FACES = 0
LAST_SFACE_CALLS = 0
LAST_RECOGNITION_BUDGET_LIMIT = 0

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

# Handles returned by os.add_dll_directory must be kept alive on Windows.
_CUDA_DLL_DIRECTORY_HANDLES = []


def configure_nvidia_cuda_dll_paths(verbose=False):
    """Add NVIDIA pip CUDA runtime DLL folders to this process on Windows.

    ONNX Runtime can list CUDAExecutionProvider even when the provider cannot be
    created because Windows cannot locate cublas/cudnn/cufft/cudart DLLs.  The
    NVIDIA pip packages install those DLLs under site-packages\nvidia\...\bin,
    but those folders are normally not in PATH.  This function adds every
    existing NVIDIA bin folder to PATH and to the DLL search path before ORT is
    imported/used.
    """
    if os.name != "nt":
        return []
    if os.environ.get("FACEATTENDANCE_DISABLE_CUDA_DLL_AUTO_PATH", "false").lower() in ("1", "true", "yes", "on"):
        return []

    candidates = []
    try:
        import site as _site
        for raw in []:
            pass
        for raw in list(getattr(_site, "getsitepackages", lambda: [])() or []):
            candidates.append(Path(raw))
        user_site = getattr(_site, "getusersitepackages", lambda: "")()
        if user_site:
            candidates.append(Path(user_site))
    except Exception:
        pass

    # Robust fallback for normal non-venv Windows installs and venvs.
    try:
        import sys as _sys
        candidates.append(Path(_sys.executable).resolve().parent / "Lib" / "site-packages")
        candidates.append(Path(getattr(_sys, "prefix", "")) / "Lib" / "site-packages")
        candidates.append(Path(getattr(_sys, "base_prefix", "")) / "Lib" / "site-packages")
    except Exception:
        pass

    subdirs = (
        "cublas", "cudnn", "cufft", "cuda_runtime", "curand", "cuda_nvrtc", "nvjitlink"
    )
    added = []
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    seen = {str(Path(part)).lower() for part in path_parts if part}

    for root in candidates:
        nvidia_root = root / "nvidia"
        for name in subdirs:
            bin_dir = nvidia_root / name / "bin"
            if not bin_dir.exists():
                continue
            key = str(bin_dir).lower()
            if key not in seen:
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                seen.add(key)
                added.append(str(bin_dir))
            try:
                handle = os.add_dll_directory(str(bin_dir))
                _CUDA_DLL_DIRECTORY_HANDLES.append(handle)
            except Exception:
                pass

    if verbose and added:
        print("[DNN] Added NVIDIA CUDA DLL paths:")
        for item in added:
            print(f"  {item}")
    return added


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


# GPU / DNN backend selection.
# IMPORTANT: the normal pip OpenCV wheels usually do not include CUDA runtime support.
# In that case the code falls back to CPU automatically and prints why.
USE_GPU_ACCELERATION = os.environ.get("FACEATTENDANCE_USE_GPU", "false").lower() in ("1", "true", "yes", "on")


def get_opencv_dnn_backend_target(prefer_gpu=None):
    """Return (backend_id, target_id, human_label) for YuNet/SFace.

    OpenCV's YuNet/SFace Python bindings can use DNN backend/target arguments
    only when the installed OpenCV build exposes those overloads. This helper
    chooses CUDA when requested and available, otherwise a safe CPU fallback.
    """
    if prefer_gpu is None:
        prefer_gpu = USE_GPU_ACCELERATION

    cpu_backend = getattr(cv2.dnn, "DNN_BACKEND_OPENCV", 3)
    cpu_target = getattr(cv2.dnn, "DNN_TARGET_CPU", 0)

    if not prefer_gpu:
        return cpu_backend, cpu_target, "CPU / OpenCV"

    cuda_backend = getattr(cv2.dnn, "DNN_BACKEND_CUDA", None)
    cuda_target = getattr(cv2.dnn, "DNN_TARGET_CUDA", None)

    if cuda_backend is None or cuda_target is None:
        return cpu_backend, cpu_target, "CPU fallback: this OpenCV build has no DNN CUDA constants"

    cuda_devices = 0
    try:
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        cuda_devices = 0

    if cuda_devices <= 0:
        return cpu_backend, cpu_target, "CPU fallback: no CUDA-enabled OpenCV device detected"

    return int(cuda_backend), int(cuda_target), f"GPU / OpenCV CUDA ({cuda_devices} device(s))"


def create_detector(input_size: Tuple[int, int] = (320, 320), prefer_gpu=None):
    backend_id, target_id, backend_label = get_opencv_dnn_backend_target(prefer_gpu)
    try:
        detector = cv2.FaceDetectorYN_create(
            str(YUNET_MODEL),
            "",
            input_size,
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
            backend_id,
            target_id,
        )
        print(f"[DNN] YuNet backend: {backend_label}")
        return detector
    except TypeError:
        print("[DNN] YuNet backend: CPU fallback because this OpenCV binding does not accept backend/target arguments")
        return cv2.FaceDetectorYN_create(
            str(YUNET_MODEL),
            "",
            input_size,
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )


def create_recognizer(prefer_gpu=None):
    if prefer_gpu is None:
        prefer_gpu = USE_GPU_ACCELERATION

    backend_key = str(SFACE_BACKEND or "auto").strip().lower()
    prefer_ort_cuda = bool(prefer_gpu) or backend_key in ("onnxruntime_cuda", "ort_cuda", "cuda")
    if should_try_onnxruntime_sface(prefer_gpu):
        try:
            return create_onnxruntime_sface_recognizer(prefer_cuda=prefer_ort_cuda)
        except Exception as exc:
            print(f"[DNN] SFace ONNX Runtime unavailable: {exc}")
            if backend_key not in ("auto", ""):
                raise

    backend_id, target_id, backend_label = get_opencv_dnn_backend_target(prefer_gpu)
    try:
        recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "", backend_id, target_id)
        print(f"[DNN] SFace backend: {backend_label}")
        return recognizer
    except TypeError:
        print("[DNN] SFace backend: CPU fallback because this OpenCV binding does not accept backend/target arguments")
        return cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")


# ============================================================
# Optional ONNX Runtime CUDA SFace backend
# ============================================================

SFACE_BACKEND = os.environ.get("FACEATTENDANCE_SFACE_BACKEND", "auto").strip().lower()
SFACE_ORT_FORCE_CPU = os.environ.get("FACEATTENDANCE_SFACE_ORT_FORCE_CPU", "false").lower() in ("1", "true", "yes", "on")
SFACE_ORT_USE_RGB = os.environ.get("FACEATTENDANCE_SFACE_ORT_USE_RGB", "false").lower() in ("1", "true", "yes", "on")

SFACE_ORT_AUTO_CLEAN_MODEL = os.environ.get("FACEATTENDANCE_SFACE_ORT_AUTO_CLEAN_MODEL", "true").lower() in ("1", "true", "yes", "on")
SFACE_ORT_CLEAN_SUFFIX = os.environ.get("FACEATTENDANCE_SFACE_ORT_CLEAN_SUFFIX", "_ort_clean")
SFACE_ORT_DYNAMIC_BATCH = os.environ.get("FACEATTENDANCE_SFACE_ORT_DYNAMIC_BATCH", "true").lower() in ("1", "true", "yes", "on")
SFACE_ORT_DYNAMIC_SUFFIX = os.environ.get("FACEATTENDANCE_SFACE_ORT_DYNAMIC_SUFFIX", "_ort_dynamic")
SFACE_ORT_LOG_VERBOSE = os.environ.get("FACEATTENDANCE_SFACE_ORT_LOG_VERBOSE", "true").lower() in ("1", "true", "yes", "on")


def _set_symbolic_batch(value_info, batch_name="N"):
    try:
        tensor_type = value_info.type.tensor_type
        shape = tensor_type.shape
        if len(shape.dim) >= 1:
            shape.dim[0].ClearField("dim_value")
            shape.dim[0].dim_param = batch_name
    except Exception:
        pass


def get_sface_ort_model_path(model_path):
    """Return an ONNX Runtime-friendly SFace model path.

    The OpenCV Zoo SFace model normally has fixed input shape [1, 3, 112, 112].
    That works for one face, but it prevents true ONNX Runtime batching. For GPU
    usage we need a dynamic batch dimension, otherwise a 4/8/16-face batch raises:
    "Got: 4 Expected: 1".

    This function creates, once, a Runtime copy that:
      1. removes weight initializers from graph inputs, reducing ORT warnings;
      2. changes the first input/output dimension to symbolic N, enabling batch.

    The original model is still used by OpenCV only for alignCrop, so existing
    face alignment behavior stays unchanged.
    """
    original = Path(model_path)
    suffix = SFACE_ORT_DYNAMIC_SUFFIX if SFACE_ORT_DYNAMIC_BATCH else SFACE_ORT_CLEAN_SUFFIX
    optimized = original.with_name(f"{original.stem}{suffix}{original.suffix}")

    if optimized.exists() and optimized.stat().st_size > 0:
        return optimized

    cleaned = original.with_name(f"{original.stem}{SFACE_ORT_CLEAN_SUFFIX}{original.suffix}")
    if not SFACE_ORT_DYNAMIC_BATCH and cleaned.exists() and cleaned.stat().st_size > 0:
        return cleaned

    if not SFACE_ORT_AUTO_CLEAN_MODEL:
        return original

    try:
        import onnx
    except Exception as exc:
        if SFACE_ORT_LOG_VERBOSE:
            print(f"[DNN] Optional ONNX optimization skipped: install onnx to create dynamic-batch model ({exc})")
        return original

    try:
        model = onnx.load(str(original))

        initializer_names = {init.name for init in model.graph.initializer}
        old_inputs = list(model.graph.input)
        new_inputs = [value_info for value_info in old_inputs if value_info.name not in initializer_names]
        removed = len(old_inputs) - len(new_inputs)
        if removed > 0:
            del model.graph.input[:]
            model.graph.input.extend(new_inputs)

        if SFACE_ORT_DYNAMIC_BATCH:
            for value_info in list(model.graph.input) + list(model.graph.output):
                _set_symbolic_batch(value_info, "N")

        onnx.checker.check_model(model)
        onnx.save(model, str(optimized))
        msg = f"[DNN] Created SFace ONNX Runtime model: {optimized}"
        msg += f" (removed {removed} initializer graph inputs"
        if SFACE_ORT_DYNAMIC_BATCH:
            msg += ", dynamic batch enabled"
        msg += ")"
        print(msg)
        return optimized
    except Exception as exc:
        print(f"[DNN] Could not create optimized SFace ONNX model, using original: {exc}")
        return original

LAST_SFACE_BATCHES = 0
LAST_SFACE_BACKEND = "OpenCV"
LAST_DETECT_MS = 0.0
LAST_SFACE_MS = 0.0
LAST_SFACE_ALIGN_MS = 0.0
LAST_SFACE_INFER_MS = 0.0
LAST_RECOGNITION_TOTAL_MS = 0.0

# Desktop/UI throttling. Rendering with PIL/Tkinter and rewriting Treeviews can
# become CPU-heavy with many unknown tracks, so the desktop app uses these
# defaults to stop the UI from becoming the bottleneck.
DESKTOP_OUTPUT_EVERY_N_FRAMES = int(float(os.environ.get("FACEATTENDANCE_DESKTOP_OUTPUT_EVERY_N_FRAMES", "2")))
DESKTOP_REPORT_EVERY_SECONDS = float(os.environ.get("FACEATTENDANCE_DESKTOP_REPORT_EVERY_SECONDS", "0.50"))


class OnnxRuntimeSFaceRecognizer:
    """SFace recognizer that keeps OpenCV only for alignCrop and runs feature()
    through ONNX Runtime.

    This is useful on normal pip OpenCV builds where cv2.cuda reports 0 devices,
    but onnxruntime-gpu exposes CUDAExecutionProvider.  The model still remains
    the same OpenCV Zoo SFace ONNX model, so existing SFace embeddings stay
    compatible when preprocessing matches OpenCV's BGR path.
    """

    def __init__(self, model_path, prefer_cuda=True, device_id=0):
        self.original_model_path = str(model_path)
        self.model_path = str(get_sface_ort_model_path(model_path))
        # Keep the original OpenCV SFace model for alignCrop. The cleaned ONNX
        # copy is only for ONNX Runtime inference.
        self.aligner = cv2.FaceRecognizerSF_create(self.original_model_path, "")
        self.backend_label = "CPU / ONNX Runtime"
        self.providers = ["CPUExecutionProvider"]
        self.input_name = None
        self.output_name = None
        self.input_width = 112
        self.input_height = 112
        self.use_rgb = bool(SFACE_ORT_USE_RGB)

        try:
            # Ensure NVIDIA pip CUDA folders are in PATH/DLL search path before
            # the CUDAExecutionProvider is created.
            try:
                configure_nvidia_cuda_dll_paths(verbose=bool(SFACE_ORT_LOG_VERBOSE))
            except Exception:
                pass
            import onnxruntime as ort
            try:
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls(directory="")
            except Exception as preload_exc:
                if SFACE_ORT_LOG_VERBOSE:
                    print(f"[DNN] ONNX Runtime CUDA DLL preload warning: {preload_exc}")
        except Exception as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install it with: py -m pip install onnxruntime-gpu"
            ) from exc

        available = list(ort.get_available_providers())
        providers = []
        if prefer_cuda and not SFACE_ORT_FORCE_CPU and "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", {
                "device_id": int(device_id),
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
            }))
            self.backend_label = "GPU / ONNX Runtime CUDA"
        elif prefer_cuda and not SFACE_ORT_FORCE_CPU:
            raise RuntimeError(
                "onnxruntime is installed, but CUDAExecutionProvider is not available. "
                f"Available providers: {available}"
            )
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        try:
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass
        try:
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            sess_options.log_severity_level = 2  # warnings and errors only
        except Exception:
            pass

        self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        self.providers = list(self.session.get_providers())
        if "CUDAExecutionProvider" in self.providers:
            self.backend_label = "GPU / ONNX Runtime CUDA"
        else:
            self.backend_label = "CPU / ONNX Runtime"

        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]
        self.input_name = input_meta.name
        self.output_name = output_meta.name

        shape = list(input_meta.shape or [])
        self.input_shape = shape
        self.supports_batch = True
        try:
            # If the model still says [1, 3, 112, 112], ONNX Runtime will reject
            # a real N-face batch with "Got: N Expected: 1". In that case we
            # fall back to a safe one-by-one loop instead of crashing.
            if len(shape) >= 1 and isinstance(shape[0], int) and int(shape[0]) == 1:
                self.supports_batch = False
        except Exception:
            self.supports_batch = True

        # SFace normally uses NCHW: [N, 3, 112, 112]. Keep safe fallbacks for
        # symbolic batch dimensions.
        try:
            if len(shape) >= 4:
                h = shape[2]
                w = shape[3]
                if isinstance(h, int) and h > 0:
                    self.input_height = int(h)
                if isinstance(w, int) and w > 0:
                    self.input_width = int(w)
        except Exception:
            pass

        if SFACE_ORT_LOG_VERBOSE:
            batch_mode = "dynamic-batch" if self.supports_batch else "fixed-batch-1"
            print(f"[DNN] SFace ORT input shape={shape}; mode={batch_mode}")

    def alignCrop(self, image, face):
        return self.aligner.alignCrop(image, face)

    def _preprocess_aligned(self, aligned):
        if aligned is None:
            raise RuntimeError("SFace alignCrop returned None")
        if aligned.shape[1] != self.input_width or aligned.shape[0] != self.input_height:
            aligned = cv2.resize(aligned, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        # OpenCV SFace embeddings are generated from BGR images.  Keep BGR by
        # default so JSON embeddings created by the previous OpenCV path remain
        # comparable.  FACEATTENDANCE_SFACE_ORT_USE_RGB=true is left as a debug
        # switch in case a custom model expects RGB.
        if self.use_rgb:
            aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        chw = aligned.astype(np.float32, copy=False).transpose(2, 0, 1)
        return chw

    def feature(self, aligned):
        batch = np.expand_dims(self._preprocess_aligned(aligned), axis=0).astype(np.float32, copy=False)
        output = self.session.run([self.output_name], {self.input_name: batch})[0]
        return np.asarray(output, dtype=np.float32).reshape(1, -1)

    def features_batch(self, image, faces):
        if not faces:
            return []

        # Safe fallback for fixed-batch SFace models. This avoids the
        # ONNXRuntimeError: Got invalid dimensions for input data, Got: N Expected: 1.
        # It is not as fast as true batching, but it keeps the app usable until the
        # dynamic-batch model is created.
        if not getattr(self, "supports_batch", True) and len(faces) > 1:
            outputs = []
            for face in faces:
                aligned = self.alignCrop(image, face)
                outputs.append(self.feature(aligned).reshape(-1))
            return outputs

        align_t0 = time.perf_counter()
        aligned_batch = []
        for face in faces:
            aligned = self.alignCrop(image, face)
            aligned_batch.append(self._preprocess_aligned(aligned))
        batch = np.stack(aligned_batch, axis=0).astype(np.float32, copy=False)
        globals()["LAST_SFACE_ALIGN_MS"] = globals().get("LAST_SFACE_ALIGN_MS", 0.0) + (time.perf_counter() - align_t0) * 1000.0
        try:
            infer_t0 = time.perf_counter()
            output = self.session.run([self.output_name], {self.input_name: batch})[0]
            globals()["LAST_SFACE_INFER_MS"] = globals().get("LAST_SFACE_INFER_MS", 0.0) + (time.perf_counter() - infer_t0) * 1000.0
        except Exception as exc:
            # Last-resort compatibility fallback for models that still contain
            # hidden fixed-batch reshape constants after input metadata was made
            # symbolic.
            if len(faces) > 1 and "Got invalid dimensions" in str(exc):
                outputs = []
                for face in faces:
                    aligned = self.alignCrop(image, face)
                    outputs.append(self.feature(aligned).reshape(-1))
                self.supports_batch = False
                print("[DNN] SFace model rejected batched input; falling back to one face per ORT call")
                return outputs
            raise
        output = np.asarray(output, dtype=np.float32)
        if output.ndim == 1:
            output = output.reshape(1, -1)
        return [output[i].reshape(-1) for i in range(output.shape[0])]


def create_onnxruntime_sface_recognizer(prefer_cuda=True):
    recognizer = OnnxRuntimeSFaceRecognizer(SFACE_MODEL, prefer_cuda=prefer_cuda)
    print(f"[DNN] SFace backend: {recognizer.backend_label} providers={recognizer.providers} model={getattr(recognizer, 'model_path', '?')}")
    return recognizer


def should_try_onnxruntime_sface(prefer_gpu=None):
    if prefer_gpu is None:
        prefer_gpu = globals().get("USE_GPU_ACCELERATION", True)
    backend = str(SFACE_BACKEND or "auto").strip().lower()
    if backend in ("opencv", "opencv_cpu", "opencv_cuda"):
        return False
    if backend in ("onnxruntime", "onnxruntime_cuda", "ort", "ort_cuda", "cuda"):
        return True
    # auto: only use ORT when the user asked for GPU acceleration.  CPU-only ORT
    # is not usually faster than OpenCV here.
    return bool(prefer_gpu)


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


def _safe_float(value, default=-1.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mark_candidate_confirmed_if_ready(name, candidate, state, now=None):
    """Mark a known face as resolved when the match is stable and very confident.

    The flag is stored in the per-person state, so once the person has passed the
    stricter check, temporary similarity drops do not make the box reappear while
    the same person is still visible. The state is reset automatically when the
    face disappears for FACE_STATE_TIMEOUT_SECONDS.
    """
    if not candidate.get("is_known", False):
        return False

    if not HIDE_CONFIRMED_KNOWN_FACES:
        return bool(state.get("confirmed_hidden", False) or state.get("recognition_stopped", False))

    stable_count = int(state.get("stable_count", 0) or 0)
    best_similarity = max(
        _safe_float(candidate.get("similarity"), -1.0),
        _safe_float(state.get("last_similarity"), -1.0),
        _safe_float(state.get("confirmed_similarity"), -1.0),
    )

    if (
        stable_count >= max(1, int(CONFIRMED_KNOWN_STABLE_FRAMES))
        and best_similarity >= float(CONFIRMED_KNOWN_SIMILARITY_THRESHOLD)
    ):
        state["confirmed_hidden"] = True
        state["confirmed_name"] = str(name)
        state["confirmed_similarity"] = best_similarity
        state["confirmed_at"] = now if now is not None else time.time()

    return bool(state.get("confirmed_hidden", False) or state.get("recognition_stopped", False))


def mark_known_stopped_if_ready(name, candidate, state, now=None):
    """Stop using a known face after a configurable number of accepted recognitions.

    This is stronger than merely hiding a confident known face: the per-person
    recognition counter is capped and future overlapping detections are treated
    as resolved skip targets. Use KNOWN_STOP_AFTER_DETECTIONS=0 to disable it.
    """
    limit = int(KNOWN_STOP_AFTER_DETECTIONS or 0)
    if limit <= 0 or not candidate.get("is_known", False):
        return False

    recognition_count = int(state.get("recognition_count", 0) or 0)
    if recognition_count >= limit:
        best_similarity = max(
            _safe_float(candidate.get("similarity"), -1.0),
            _safe_float(state.get("last_similarity"), -1.0),
            _safe_float(state.get("confirmed_similarity"), -1.0),
        )
        state["recognition_count"] = limit
        state["recognition_stopped"] = True
        state["confirmed_hidden"] = True
        state["confirmed_name"] = str(name)
        state["confirmed_similarity"] = best_similarity
        state["stopped_at"] = now if now is not None else time.time()
        candidate["known_recognition_stopped"] = True

    return bool(state.get("recognition_stopped", False))


def is_candidate_confirmed_hidden(candidate, face_state_by_name):
    if not is_candidate_accepted_known(candidate):
        return False

    name = str(candidate.get("name", ""))
    state = face_state_by_name.get(name) if face_state_by_name is not None else None
    return bool(state and (state.get("confirmed_hidden", False) or state.get("recognition_stopped", False)))


def _descriptor_from_candidate(candidate):
    descriptor = candidate.get("descriptor")
    if descriptor is None:
        return None
    try:
        arr = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def _new_unknown_track_id(unknown_state_by_id):
    next_number = int(unknown_state_by_id.get("_next_number", 1))
    unknown_state_by_id["_next_number"] = next_number + 1
    return f"unknown:{next_number}"


def _find_unknown_track(descriptor, candidate_box, unknown_state_by_id, now=None):
    best_track_id = None
    best_score = -1.0
    for track_id, state in unknown_state_by_id.items():
        if str(track_id).startswith("_") or not isinstance(state, dict):
            continue
        avg_descriptor = state.get("avg_descriptor")
        if avg_descriptor is None:
            continue
        try:
            avg = np.asarray(avg_descriptor, dtype=np.float32).reshape(-1)
        except Exception:
            continue
        if avg.size != descriptor.size or not np.all(np.isfinite(avg)):
            continue
        avg_norm = float(np.linalg.norm(avg))
        if avg_norm <= 1e-8:
            continue
        score = cosine_similarity(descriptor, avg / avg_norm)
        if score > best_score:
            best_score = score
            best_track_id = track_id

    if best_track_id is not None and best_score >= float(UNKNOWN_TRACK_MATCH_THRESHOLD):
        return best_track_id, unknown_state_by_id[best_track_id], best_score

    return None, None, best_score



def is_candidate_accepted_known(candidate):
    """True only when a known label passed the temporal identity voter."""
    if not candidate or not candidate.get("is_known", False):
        return False
    if not bool(IDENTITY_VOTING_ENABLED):
        return True
    return bool(candidate.get("identity_confirmed", False))


def is_candidate_pending_identity(candidate):
    return bool(
        candidate
        and candidate.get("is_known", False)
        and bool(IDENTITY_VOTING_ENABLED)
        and not candidate.get("identity_confirmed", False)
    )


def _identity_vote_track_id(identity_state_by_track):
    next_number = int(identity_state_by_track.get("_next_number", 1))
    identity_state_by_track["_next_number"] = next_number + 1
    return f"identity:{next_number}"


def _candidate_descriptor(candidate):
    descriptor = candidate.get("descriptor")
    if descriptor is None:
        return None
    try:
        arr = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def _find_identity_vote_track(candidate, identity_state_by_track, now=None, excluded_track_ids=None):
    box = candidate.get("box")
    descriptor = _candidate_descriptor(candidate)
    best_id = None
    best_score = -1.0

    excluded_track_ids = set(excluded_track_ids or set())
    for track_id, state in (identity_state_by_track or {}).items():
        if str(track_id).startswith("_") or track_id in excluded_track_ids or not isinstance(state, dict):
            continue
        if now is not None and now - float(state.get("last_seen", 0.0) or 0.0) > float(IDENTITY_TRACK_TIMEOUT_SECONDS):
            continue

        track_box = state.get("last_box")
        if track_box is None or box is None:
            continue

        iou = intersection_over_union(box, track_box)
        center_ratio = center_distance_ratio(box, track_box)
        spatial_match = iou >= float(IDENTITY_TRACK_IOU_THRESHOLD) or center_ratio <= float(IDENTITY_TRACK_CENTER_RATIO)

        desc_score = -1.0
        avg_descriptor = state.get("avg_descriptor")
        if descriptor is not None and avg_descriptor is not None:
            try:
                avg = np.asarray(avg_descriptor, dtype=np.float32).reshape(-1)
                avg_norm = float(np.linalg.norm(avg))
                if avg.size == descriptor.size and avg_norm > 1e-8:
                    desc_score = cosine_similarity(descriptor, avg / avg_norm)
            except Exception:
                desc_score = -1.0

        descriptor_match = desc_score >= float(IDENTITY_TRACK_DESCRIPTOR_THRESHOLD)
        if not spatial_match and not descriptor_match:
            continue

        # Keep spatial continuity as the primary tracker signal. Descriptor score
        # is a fallback for small box jitter or short occlusion.
        score = max(float(iou), 1.0 - float(center_ratio), float(desc_score))
        if score > best_score:
            best_score = score
            best_id = track_id

    return best_id


def _new_identity_track(candidate, identity_state_by_track, now=None):
    track_id = _identity_vote_track_id(identity_state_by_track)
    descriptor = _candidate_descriptor(candidate)
    identity_state_by_track[track_id] = {
        "last_seen": now if now is not None else time.time(),
        "last_box": candidate.get("box"),
        "avg_descriptor": descriptor.tolist() if descriptor is not None else None,
        "votes": deque(maxlen=max(3, int(IDENTITY_VOTE_WINDOW))),
        "confirmed_name": None,
        "confirmed_similarity": -1.0,
        "confirmed_at": 0.0,
        "last_vote_summary": {},
    }
    return track_id


def _update_identity_track_descriptor(state, descriptor):
    if descriptor is None:
        return
    avg_descriptor = state.get("avg_descriptor")
    if avg_descriptor is None:
        state["avg_descriptor"] = descriptor.tolist()
        return
    try:
        avg = np.asarray(avg_descriptor, dtype=np.float32).reshape(-1)
        if avg.size != descriptor.size:
            state["avg_descriptor"] = descriptor.tolist()
            return
        alpha = min(1.0, max(0.0, float(IDENTITY_TRACK_DESCRIPTOR_ALPHA)))
        avg = l2_normalize((1.0 - alpha) * avg + alpha * descriptor)
        state["avg_descriptor"] = avg.tolist()
    except Exception:
        state["avg_descriptor"] = descriptor.tolist()


def _known_vote_is_usable(vote):
    name = vote.get("name")
    if not name or name == UNKNOWN_LABEL:
        return False
    if float(vote.get("similarity", -1.0) or -1.0) < float(IDENTITY_MIN_ACCEPT_SIMILARITY):
        return False
    if float(vote.get("margin", 0.0) or 0.0) < float(IDENTITY_MIN_ACCEPT_MARGIN):
        return False
    return True


def _identity_vote_summary(votes):
    votes = list(votes or [])
    usable = [vote for vote in votes if _known_vote_is_usable(vote)]
    counts = Counter(str(vote.get("name")) for vote in usable)
    if not counts:
        return {
            "top_name": None,
            "top_count": 0,
            "second_count": 0,
            "total_votes": len(votes),
            "usable_votes": 0,
            "ratio": 0.0,
            "avg_similarity": -1.0,
            "avg_margin": 0.0,
        }

    ordered = counts.most_common()
    top_name, top_count = ordered[0]
    second_count = ordered[1][1] if len(ordered) >= 2 else 0
    top_votes = [vote for vote in usable if str(vote.get("name")) == top_name]
    avg_similarity = float(np.mean([float(vote.get("similarity", -1.0) or -1.0) for vote in top_votes])) if top_votes else -1.0
    avg_margin = float(np.mean([float(vote.get("margin", 0.0) or 0.0) for vote in top_votes])) if top_votes else 0.0
    # Denominator is the full recent window, not only usable known votes.
    # Unknown/ambiguous frames therefore act as negative evidence instead of
    # letting four isolated false positives confirm a person.
    ratio = float(top_count) / float(max(1, len(votes)))
    return {
        "top_name": top_name,
        "top_count": int(top_count),
        "second_count": int(second_count),
        "total_votes": len(votes),
        "usable_votes": len(usable),
        "ratio": ratio,
        "avg_similarity": avg_similarity,
        "avg_margin": avg_margin,
    }


def _identity_summary_passes(summary, switch=False):
    if not summary or not summary.get("top_name"):
        return False
    min_votes = int(IDENTITY_SWITCH_MIN_VOTES if switch else IDENTITY_MIN_KNOWN_VOTES)
    min_ratio = float(IDENTITY_SWITCH_MIN_RATIO if switch else IDENTITY_MIN_KNOWN_VOTE_RATIO)
    min_margin_votes = int(IDENTITY_SWITCH_MARGIN_VOTES if switch else IDENTITY_MIN_KNOWN_MARGIN_VOTES)
    return (
        int(summary.get("top_count", 0) or 0) >= max(1, min_votes)
        and float(summary.get("ratio", 0.0) or 0.0) >= min_ratio
        and (int(summary.get("top_count", 0) or 0) - int(summary.get("second_count", 0) or 0)) >= min_margin_votes
        and float(summary.get("avg_similarity", -1.0) or -1.0) >= float(IDENTITY_MIN_ACCEPT_SIMILARITY)
    )


def _apply_identity_confirmation_to_candidate(candidate, state, summary):
    confirmed_name = state.get("confirmed_name")
    if not confirmed_name:
        candidate["identity_confirmed"] = False
        candidate["identity_vote_count"] = int(summary.get("top_count", 0) or 0)
        candidate["identity_vote_ratio"] = float(summary.get("ratio", 0.0) or 0.0)
        candidate["identity_top_name"] = summary.get("top_name")
        return candidate

    candidate["identity_confirmed"] = True
    candidate["identity_top_name"] = confirmed_name
    candidate["identity_vote_count"] = int(summary.get("top_count", 0) or 0)
    candidate["identity_vote_ratio"] = float(summary.get("ratio", 0.0) or 0.0)
    candidate["name"] = confirmed_name
    candidate["is_known"] = True
    # Use the strongest value available for status/reporting, but keep raw_* for diagnostics.
    candidate["similarity"] = max(
        _safe_float(candidate.get("similarity"), -1.0),
        _safe_float(state.get("confirmed_similarity"), -1.0),
        _safe_float(summary.get("avg_similarity"), -1.0),
    )
    return candidate


def update_identity_vote_tracking(candidates, identity_state_by_track, now=None):
    """Confirm known labels by temporal voting per physical face.

    A single physical face is tracked by box continuity plus descriptor fallback.
    A name becomes reportable only when it wins several recent votes. Temporary
    wrong names are kept as pending diagnostics and are not counted in Known
    persons, attendance, or stop counters.
    """
    if not bool(IDENTITY_VOTING_ENABLED):
        for candidate in candidates or []:
            if candidate.get("is_known", False):
                candidate["identity_confirmed"] = True
        return candidates

    if identity_state_by_track is None:
        return candidates
    if now is None:
        now = time.time()

    # Drop stale tracks before matching new candidates.
    for track_id in list(identity_state_by_track.keys()):
        if str(track_id).startswith("_"):
            continue
        state = identity_state_by_track.get(track_id)
        if not isinstance(state, dict):
            continue
        if now - float(state.get("last_seen", 0.0) or 0.0) > float(IDENTITY_TRACK_TIMEOUT_SECONDS):
            del identity_state_by_track[track_id]

    assigned_track_ids = set()
    for candidate in candidates or []:
        if candidate.get("resolved_skip", False) and candidate.get("is_known", False):
            candidate["identity_confirmed"] = True
            continue

        track_id = _find_identity_vote_track(candidate, identity_state_by_track, now, excluded_track_ids=assigned_track_ids)
        if track_id is not None:
            assigned_track_ids.add(track_id)
        if track_id is None:
            track_id = _new_identity_track(candidate, identity_state_by_track, now)
            assigned_track_ids.add(track_id)
        state = identity_state_by_track.get(track_id)
        if not isinstance(state, dict):
            continue

        candidate["identity_track_id"] = track_id
        state["last_seen"] = now
        state["last_box"] = candidate.get("box")
        _update_identity_track_descriptor(state, _candidate_descriptor(candidate))

        raw_name = str(candidate.get("name") or UNKNOWN_LABEL)
        raw_similarity = _safe_float(candidate.get("similarity"), -1.0)
        raw_margin = _safe_float(candidate.get("similarity_margin"), 0.0)
        candidate["raw_name"] = raw_name
        candidate["raw_similarity"] = raw_similarity

        # Unknown/raw-low votes remain in the window as negative evidence; they
        # prevent a person from being accepted after one lucky high score.
        vote_name = raw_name if candidate.get("is_known", False) else UNKNOWN_LABEL
        state.setdefault("votes", deque(maxlen=max(3, int(IDENTITY_VOTE_WINDOW))))
        if not isinstance(state["votes"], deque):
            state["votes"] = deque(list(state["votes"]), maxlen=max(3, int(IDENTITY_VOTE_WINDOW)))
        state["votes"].append({
            "name": vote_name,
            "similarity": raw_similarity,
            "margin": raw_margin,
            "time": now,
        })

        summary = _identity_vote_summary(state.get("votes"))
        state["last_vote_summary"] = summary
        confirmed_name = state.get("confirmed_name")

        if confirmed_name:
            top_name = summary.get("top_name")
            if top_name and str(top_name) != str(confirmed_name) and _identity_summary_passes(summary, switch=True):
                # A real track can switch when a person leaves and another enters
                # in almost the same position. Require a stronger vote to switch.
                state["confirmed_name"] = str(top_name)
                state["confirmed_similarity"] = float(summary.get("avg_similarity", raw_similarity) or raw_similarity)
                state["confirmed_at"] = now
        else:
            if _identity_summary_passes(summary, switch=False):
                state["confirmed_name"] = str(summary.get("top_name"))
                state["confirmed_similarity"] = float(summary.get("avg_similarity", raw_similarity) or raw_similarity)
                state["confirmed_at"] = now

        _apply_identity_confirmation_to_candidate(candidate, state, summary)

    return candidates


def update_unknown_tracking(candidates, unknown_state_by_id, now=None):
    """Assign stable track ids to Unknown faces and hide repeated ones.

    This is intentionally separate from Moodle/desktop review storage. It only
    controls what remains visible/unresolved in the live camera UI.
    """
    if unknown_state_by_id is None:
        return
    if now is None:
        now = time.time()

    for candidate in candidates:
        if candidate.get("is_known", False):
            continue

        descriptor = _descriptor_from_candidate(candidate)
        if descriptor is None:
            # A hidden/resolved Unknown can be carried forward by box match
            # without running SFace again. Keep the track alive so cleanup does
            # not make it reappear after FACE_STATE_TIMEOUT_SECONDS.
            track_id = candidate.get("unknown_track_id")
            state = unknown_state_by_id.get(track_id) if track_id else None
            if isinstance(state, dict) and state.get("confirmed_hidden", False):
                state["last_seen"] = now
                state["last_box"] = candidate.get("box", state.get("last_box"))
                candidate["unknown_stable_count"] = int(state.get("stable_count", 0) or 0)
                candidate["unknown_confirmed_hidden"] = True
            continue

        track_id, state, match_score = _find_unknown_track(descriptor, candidate.get("box"), unknown_state_by_id, now)
        if state is None:
            track_id = _new_unknown_track_id(unknown_state_by_id)
            state = {
                "stable_count": 0,
                "last_seen": 0.0,
                "last_box": None,
                "avg_descriptor": descriptor.astype(float).tolist(),
                "confirmed_hidden": False,
                "confirmed_at": 0.0,
                "best_track_similarity": -1.0,
            }
            unknown_state_by_id[track_id] = state

        state["stable_count"] = int(state.get("stable_count", 0) or 0) + 1
        state["last_seen"] = now
        state["last_box"] = candidate.get("box")
        state["best_track_similarity"] = max(
            _safe_float(state.get("best_track_similarity"), -1.0),
            _safe_float(match_score, -1.0),
        )

        old_avg = _descriptor_from_candidate({"descriptor": state.get("avg_descriptor")})
        if old_avg is None:
            new_avg = descriptor
        else:
            alpha = min(1.0, max(0.0, float(UNKNOWN_TRACK_DESCRIPTOR_ALPHA)))
            new_avg = old_avg * (1.0 - alpha) + descriptor * alpha
            norm = float(np.linalg.norm(new_avg))
            if norm > 1e-8:
                new_avg = new_avg / norm
            else:
                new_avg = descriptor
        state["avg_descriptor"] = new_avg.astype(float).tolist()

        if (
            HIDE_CONFIRMED_UNKNOWN_FACES
            and int(state.get("stable_count", 0) or 0) >= max(1, int(CONFIRMED_UNKNOWN_STABLE_FRAMES))
        ):
            state["confirmed_hidden"] = True
            state["confirmed_at"] = now

        candidate["unknown_track_id"] = track_id
        candidate["unknown_stable_count"] = int(state.get("stable_count", 0) or 0)
        candidate["unknown_confirmed_hidden"] = bool(state.get("confirmed_hidden", False))


def cleanup_unknown_tracking(unknown_state_by_id, now=None):
    if unknown_state_by_id is None:
        return
    if now is None:
        now = time.time()
    for track_id in list(unknown_state_by_id.keys()):
        if str(track_id).startswith("_"):
            continue
        state = unknown_state_by_id.get(track_id)
        if not isinstance(state, dict):
            continue
        if now - float(state.get("last_seen", 0.0) or 0.0) > FACE_STATE_TIMEOUT_SECONDS:
            del unknown_state_by_id[track_id]


def is_candidate_confirmed_unknown_hidden(candidate, unknown_state_by_id):
    if not HIDE_CONFIRMED_UNKNOWN_FACES or candidate.get("is_known", False):
        return False
    if bool(candidate.get("unknown_confirmed_hidden", False)):
        return True
    track_id = candidate.get("unknown_track_id")
    state = unknown_state_by_id.get(track_id) if unknown_state_by_id is not None and track_id else None
    return bool(state and state.get("confirmed_hidden", False))


def is_candidate_resolved_hidden(candidate, face_state_by_name, unknown_state_by_id=None):
    if is_candidate_accepted_known(candidate):
        return is_candidate_confirmed_hidden(candidate, face_state_by_name)
    if is_candidate_pending_identity(candidate):
        return False
    return is_candidate_confirmed_unknown_hidden(candidate, unknown_state_by_id)


def build_resolved_face_skip_targets(face_state_by_name=None, unknown_state_by_id=None):
    """Return confirmed/hidden faces that can skip SFace recognition.

    Each target is still visible to the detector, but we already know that it is
    resolved for UI purposes. Reusing its existing state prevents repeatedly
    running SFace on people who have already been handled.
    """
    targets = []

    for name, state in (face_state_by_name or {}).items():
        if not isinstance(state, dict):
            continue
        if not (state.get("confirmed_hidden", False) or state.get("recognition_stopped", False)):
            continue
        box = state.get("last_box")
        if box is None:
            continue
        targets.append({
            "kind": "known",
            "name": name,
            "box": box,
            "similarity": _safe_float(state.get("confirmed_similarity", state.get("last_similarity", -1.0)), -1.0),
            "recognition_stopped": bool(state.get("recognition_stopped", False)),
            "recognition_count": int(state.get("recognition_count", 0) or 0),
        })

    for track_id, state in (unknown_state_by_id or {}).items():
        if str(track_id).startswith("_") or not isinstance(state, dict):
            continue
        if not state.get("confirmed_hidden", False):
            continue
        box = state.get("last_box")
        if box is None:
            continue
        targets.append({
            "kind": "unknown",
            "track_id": track_id,
            "box": box,
            "stable_count": int(state.get("stable_count", 0) or 0),
            "avg_descriptor": state.get("avg_descriptor"),
        })

    return targets


def find_resolved_face_skip_target(candidate_box, resolved_skip_targets):
    if not SKIP_RESOLVED_FACE_RECOGNITION or not resolved_skip_targets:
        return None

    best_target = None
    best_score = -1.0
    for target in resolved_skip_targets:
        target_box = target.get("box")
        if target_box is None:
            continue

        iou = intersection_over_union(candidate_box, target_box)
        center_ratio = center_distance_ratio(candidate_box, target_box)
        matched = iou >= float(RESOLVED_FACE_SKIP_IOU) or center_ratio <= float(RESOLVED_FACE_SKIP_CENTER_RATIO)
        if not matched:
            continue

        # Higher is better. Prefer stronger box overlap; center match is fallback.
        score = max(float(iou), 1.0 - float(center_ratio))
        if score > best_score:
            best_score = score
            best_target = target

    return best_target


def make_candidate_from_resolved_target(original_box, area, source, search_zoom, target):
    if target.get("kind") == "known":
        return {
            "box": original_box,
            "name": target.get("name", UNKNOWN_LABEL),
            "is_known": True,
            "similarity": _safe_float(target.get("similarity"), -1.0),
            "area": area,
            "source": source,
            "search_zoom": search_zoom,
            "resolved_skip": True,
            "identity_confirmed": True,
            "known_recognition_stopped": bool(target.get("recognition_stopped", False)),
            "known_recognition_count": int(target.get("recognition_count", 0) or 0),
        }

    return {
        "box": original_box,
        "name": UNKNOWN_LABEL,
        "is_known": False,
        "similarity": -1.0,
        "area": area,
        "source": source,
        "search_zoom": search_zoom,
        "resolved_skip": True,
        "unknown_track_id": target.get("track_id"),
        "unknown_stable_count": int(target.get("stable_count", 0) or 0),
        "unknown_confirmed_hidden": True,
    }


def apply_recognition_budget(faces, frame_number=0):
    """Optionally cap how many non-resolved faces are passed through SFace.

    This is disabled by default. When enabled, the selected subset rotates across
    frames so small/far faces are not permanently starved by the largest faces.
    """
    limit = int(MAX_RECOGNITIONS_PER_FRAME or 0)
    if limit <= 0 or len(faces) <= limit:
        return list(faces)

    faces = list(faces)
    start = (int(frame_number or 0) * limit) % len(faces)
    rotated = faces[start:] + faces[:start]
    return rotated[:limit]


def split_candidates_by_confirmation(candidates, face_state_by_name, unknown_state_by_id=None):
    visible = []
    hidden_confirmed = []
    for candidate in candidates:
        if is_candidate_resolved_hidden(candidate, face_state_by_name, unknown_state_by_id):
            hidden_confirmed.append(candidate)
        else:
            visible.append(candidate)
    return visible, hidden_confirmed


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


def extract_features_batch(recognizer, image, faces):
    """Extract SFace descriptors for a list of faces.

    ONNX Runtime supports a real batch path, which is critical for GPU usage.
    The OpenCV fallback keeps the previous behavior.
    """
    global LAST_SFACE_BATCHES, LAST_SFACE_BACKEND, LAST_SFACE_ALIGN_MS, LAST_SFACE_INFER_MS
    faces = list(faces or [])
    if not faces:
        return []

    if hasattr(recognizer, "features_batch"):
        LAST_SFACE_BATCHES += 1
        LAST_SFACE_BACKEND = getattr(recognizer, "backend_label", "ONNX Runtime")
        return [l2_normalize(feature) for feature in recognizer.features_batch(image, faces)]

    features = []
    LAST_SFACE_BACKEND = "OpenCV"
    for face in faces:
        LAST_SFACE_BATCHES += 1
        t0 = time.perf_counter()
        features.append(extract_feature(recognizer, image, face))
        LAST_SFACE_ALIGN_MS += (time.perf_counter() - t0) * 1000.0
    LAST_SFACE_INFER_MS = 0.0
    return features


_KNOWN_FEATURE_INDEX_CACHE = {"signature": None, "index": None}


def _known_features_signature(known_features):
    """Cheap cache signature that survives per-frame shallow dict copies."""
    entries = []
    for person_name, features in sorted((known_features or {}).items(), key=lambda item: str(item[0])):
        feature_ids = tuple(id(feature) for feature in features)
        entries.append((str(person_name), len(features), feature_ids))
    return tuple(entries)


def build_known_feature_index(known_features):
    names = []
    vectors = []

    for person_name, features in (known_features or {}).items():
        for known_feature in features:
            descriptor = l2_normalize(np.asarray(known_feature, dtype=np.float32).reshape(-1))
            if descriptor.size == 0:
                continue
            names.append(person_name)
            vectors.append(descriptor)

    if not vectors:
        return {"names": [], "matrix": np.empty((0, 0), dtype=np.float32)}

    return {
        "names": names,
        "matrix": np.vstack(vectors).astype(np.float32, copy=False),
    }


def get_known_feature_index(known_features):
    signature = _known_features_signature(known_features)
    if _KNOWN_FEATURE_INDEX_CACHE.get("signature") != signature:
        _KNOWN_FEATURE_INDEX_CACHE["signature"] = signature
        _KNOWN_FEATURE_INDEX_CACHE["index"] = build_known_feature_index(known_features)
    return _KNOWN_FEATURE_INDEX_CACHE.get("index")


def recognize_feature_detailed(feature, known_features):
    """Return top-match diagnostics for SFace recognition.

    The older code only returned the best name and immediately accepted that
    name when it was above SFACE_SIMILARITY_THRESHOLD. With many enrolled
    persons, a single noisy frame may briefly cross that threshold. Returning
    the second-best score and the score margin lets the temporal voter reject
    weak/ambiguous one-frame labels before they reach the Known-person report.
    """
    if isinstance(known_features, dict) and "matrix" in known_features and "names" in known_features:
        index = known_features
    else:
        index = get_known_feature_index(known_features)
    if not index or index["matrix"].size == 0:
        return {
            "name": UNKNOWN_LABEL,
            "similarity": -1.0,
            "best_name": UNKNOWN_LABEL,
            "best_similarity": -1.0,
            "second_name": UNKNOWN_LABEL,
            "second_similarity": -1.0,
            "margin": 0.0,
            "is_known": False,
        }

    query = l2_normalize(np.asarray(feature, dtype=np.float32).reshape(-1))
    scores = index["matrix"].dot(query)
    if scores.size == 0:
        return {
            "name": UNKNOWN_LABEL,
            "similarity": -1.0,
            "best_name": UNKNOWN_LABEL,
            "best_similarity": -1.0,
            "second_name": UNKNOWN_LABEL,
            "second_similarity": -1.0,
            "margin": 0.0,
            "is_known": False,
        }

    best_idx = int(np.argmax(scores))
    best_similarity = float(scores[best_idx])
    best_name = index["names"][best_idx]

    # Compute the best score belonging to a different person. This is more useful
    # than the second embedding, because one person can have multiple samples.
    second_name = UNKNOWN_LABEL
    second_similarity = -1.0
    for idx in np.argsort(scores)[::-1]:
        candidate_name = index["names"][int(idx)]
        if str(candidate_name) == str(best_name):
            continue
        second_name = candidate_name
        second_similarity = float(scores[int(idx)])
        break

    margin = best_similarity - second_similarity if second_similarity > -0.5 else best_similarity
    is_known = best_similarity >= float(SFACE_SIMILARITY_THRESHOLD)
    name = best_name if is_known else UNKNOWN_LABEL

    return {
        "name": name,
        "similarity": best_similarity,
        "best_name": best_name,
        "best_similarity": best_similarity,
        "second_name": second_name,
        "second_similarity": second_similarity,
        "margin": float(margin),
        "is_known": bool(is_known),
    }


def recognize_feature(feature, known_features):
    """Backward-compatible best-match helper."""
    result = recognize_feature_detailed(feature, known_features)
    return result["name"], result["similarity"]


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


def detect_faces_with_search_zoom(
    detector,
    recognizer,
    known_features,
    frame,
    full_grid=False,
    last_box=None,
    locked_name=None,
    resolved_skip_targets=None,
    frame_number=0,
):
    global LAST_DETECTED_FACES, LAST_SFACE_CALLS, LAST_RECOGNITION_BUDGET_LIMIT, LAST_SFACE_BATCHES, LAST_SFACE_BACKEND
    global LAST_DETECT_MS, LAST_SFACE_MS, LAST_SFACE_ALIGN_MS, LAST_SFACE_INFER_MS, LAST_RECOGNITION_TOTAL_MS
    total_t0 = time.perf_counter()
    LAST_DETECTED_FACES = 0
    LAST_SFACE_CALLS = 0
    LAST_SFACE_BATCHES = 0
    LAST_DETECT_MS = 0.0
    LAST_SFACE_MS = 0.0
    LAST_SFACE_ALIGN_MS = 0.0
    LAST_SFACE_INFER_MS = 0.0
    LAST_RECOGNITION_TOTAL_MS = 0.0
    LAST_SFACE_BACKEND = getattr(recognizer, "backend_label", "OpenCV")
    LAST_RECOGNITION_BUDGET_LIMIT = int(MAX_RECOGNITIONS_PER_FRAME or 0)
    """Detect faces and recognize only unresolved candidates.

    Crowded-room optimization:
      - YuNet still detects the full frame.
      - Detections that overlap confirmed/hidden known or unknown tracks reuse the
        existing state and skip SFace extraction.
      - Optional MAX_RECOGNITIONS_PER_FRAME can cap remaining SFace calls.
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
            base_image = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            base_to_original_x = crop_w / float(frame_w)
            base_to_original_y = crop_h / float(frame_h)

        recognition_input, det_to_base_x, det_to_base_y = resize_for_recognition(base_image)

        detect_t0 = time.perf_counter()
        faces = detect_faces(detector, recognition_input)
        LAST_DETECT_MS += (time.perf_counter() - detect_t0) * 1000.0
        LAST_DETECTED_FACES += int(len(faces))

        if len(faces) == 0:
            continue

        faces = sorted(faces, key=face_area, reverse=True)
        unresolved_faces = []

        for face in faces:
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

            target = find_resolved_face_skip_target(original_box, resolved_skip_targets)
            if target is not None:
                candidates.append(make_candidate_from_resolved_target(original_box, area, source, search_zoom, target))
                continue

            unresolved_faces.append((face, original_box, area))

        budgeted_faces = list(apply_recognition_budget(unresolved_faces, frame_number=frame_number))
        if budgeted_faces:
            face_inputs = [item[0] for item in budgeted_faces]
            sface_t0 = time.perf_counter()
            features = extract_features_batch(recognizer, recognition_input, face_inputs)
            LAST_SFACE_MS += (time.perf_counter() - sface_t0) * 1000.0
            LAST_SFACE_CALLS += len(features)

            for (face, original_box, area), feature in zip(budgeted_faces, features):
                match = recognize_feature_detailed(feature, known_features)
                name = match["name"]
                similarity = match["similarity"]
                is_known = bool(match.get("is_known", False)) and name != UNKNOWN_LABEL

                candidates.append({
                    "box": original_box,
                    "name": name,
                    "is_known": is_known,
                    "similarity": similarity,
                    "second_name": match.get("second_name", UNKNOWN_LABEL),
                    "second_similarity": match.get("second_similarity", -1.0),
                    "similarity_margin": match.get("margin", 0.0),
                    "raw_name": name,
                    "raw_similarity": similarity,
                    "descriptor": feature.reshape(-1).astype(float).tolist(),
                    "area": area,
                    "source": source,
                    "search_zoom": search_zoom,
                })

    LAST_RECOGNITION_TOTAL_MS = (time.perf_counter() - total_t0) * 1000.0
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
    global HIDE_CONFIRMED_KNOWN_FACES, CONFIRMED_KNOWN_SIMILARITY_THRESHOLD, CONFIRMED_KNOWN_STABLE_FRAMES
    global HIDE_CONFIRMED_UNKNOWN_FACES, CONFIRMED_UNKNOWN_STABLE_FRAMES, UNKNOWN_TRACK_MATCH_THRESHOLD
    global SKIP_RESOLVED_FACE_RECOGNITION, RESOLVED_FACE_SKIP_IOU, RESOLVED_FACE_SKIP_CENTER_RATIO, MAX_RECOGNITIONS_PER_FRAME
    global KNOWN_STOP_AFTER_DETECTIONS
    global SFACE_BACKEND

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
    parser.add_argument("--confirmed-similarity", type=float, default=None, help="Similarity needed before a stable known face is hidden from the live view. Example: 0.52")
    parser.add_argument("--confirmed-stable-frames", type=int, default=None, help="Stable frames needed before a high-confidence known face is hidden.")
    parser.add_argument("--show-confirmed", action="store_true", help="Keep drawing confirmed known faces instead of hiding their boxes.")
    parser.add_argument("--confirmed-unknown-frames", type=int, default=None, help="Hide the same Unknown face after this many repeated detections. Example: 10")
    parser.add_argument("--show-confirmed-unknown", action="store_true", help="Keep drawing repeated Unknown faces instead of hiding them.")
    parser.add_argument("--no-skip-resolved-recognition", action="store_true", help="Do not skip SFace recognition for already hidden/resolved faces.")
    parser.add_argument("--max-recognitions-per-frame", type=int, default=None, help="Cap SFace recognitions per frame. 0 means unlimited. Example for crowded rooms: 18")
    parser.add_argument("--known-stop-after-detections", type=int, default=None, help="Stop counting/using a known person after this many accepted recognitions. 0 disables. Example: 150")
    parser.add_argument("--sface-backend", default=None, help="SFace backend: auto, opencv, onnxruntime_cuda, or ort_cuda")
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

    HIDE_CONFIRMED_KNOWN_FACES = bool_config_value(config.get("hide_confirmed_known_faces", HIDE_CONFIRMED_KNOWN_FACES), HIDE_CONFIRMED_KNOWN_FACES)
    if args.show_confirmed:
        HIDE_CONFIRMED_KNOWN_FACES = False
    CONFIRMED_KNOWN_SIMILARITY_THRESHOLD = float(pick("confirmed_similarity_threshold", args.confirmed_similarity, CONFIRMED_KNOWN_SIMILARITY_THRESHOLD))
    CONFIRMED_KNOWN_STABLE_FRAMES = int(float(pick("confirmed_stable_frames", args.confirmed_stable_frames, CONFIRMED_KNOWN_STABLE_FRAMES)))

    HIDE_CONFIRMED_UNKNOWN_FACES = bool_config_value(config.get("hide_confirmed_unknown_faces", HIDE_CONFIRMED_UNKNOWN_FACES), HIDE_CONFIRMED_UNKNOWN_FACES)
    if args.show_confirmed_unknown:
        HIDE_CONFIRMED_UNKNOWN_FACES = False
    CONFIRMED_UNKNOWN_STABLE_FRAMES = int(float(pick("confirmed_unknown_frames", args.confirmed_unknown_frames, CONFIRMED_UNKNOWN_STABLE_FRAMES)))
    UNKNOWN_TRACK_MATCH_THRESHOLD = float(config.get("unknown_track_match_threshold", UNKNOWN_TRACK_MATCH_THRESHOLD))

    SKIP_RESOLVED_FACE_RECOGNITION = bool_config_value(config.get("skip_resolved_recognition", SKIP_RESOLVED_FACE_RECOGNITION), SKIP_RESOLVED_FACE_RECOGNITION)
    if args.no_skip_resolved_recognition:
        SKIP_RESOLVED_FACE_RECOGNITION = False
    RESOLVED_FACE_SKIP_IOU = float(config.get("resolved_skip_iou", RESOLVED_FACE_SKIP_IOU))
    RESOLVED_FACE_SKIP_CENTER_RATIO = float(config.get("resolved_skip_center_ratio", RESOLVED_FACE_SKIP_CENTER_RATIO))
    MAX_RECOGNITIONS_PER_FRAME = int(float(pick("max_recognitions_per_frame", args.max_recognitions_per_frame, MAX_RECOGNITIONS_PER_FRAME)))
    KNOWN_STOP_AFTER_DETECTIONS = int(float(pick("known_stop_after_detections", args.known_stop_after_detections, KNOWN_STOP_AFTER_DETECTIONS)))
    SFACE_BACKEND = str(pick("sface_backend", args.sface_backend, SFACE_BACKEND)).strip().lower()

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
    print(f"  Manual zoom: {manual_zoom_status_text()}")
    print(f"  Hide confirmed known faces: {HIDE_CONFIRMED_KNOWN_FACES}")
    print(f"  Hide repeated unknown faces: {HIDE_CONFIRMED_UNKNOWN_FACES} after {CONFIRMED_UNKNOWN_STABLE_FRAMES} detection(s)")
    print(f"  Confirmed similarity threshold: {CONFIRMED_KNOWN_SIMILARITY_THRESHOLD:.2f}")
    print(f"  Confirmed stable frames: {CONFIRMED_KNOWN_STABLE_FRAMES}")
    print(f"  Skip recognition for resolved faces: {SKIP_RESOLVED_FACE_RECOGNITION}")
    print(f"  Max SFace recognitions per frame: {MAX_RECOGNITIONS_PER_FRAME if MAX_RECOGNITIONS_PER_FRAME > 0 else 'unlimited'}")
    print(f"  Stop known after detections: {KNOWN_STOP_AFTER_DETECTIONS if KNOWN_STOP_AFTER_DETECTIONS > 0 else 'disabled'}")
    print(f"  SFace backend: {SFACE_BACKEND}\n")

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
        safe_url = re.sub(r"secret=[^&]+", "secret=<hidden>", url)
        raise RuntimeError(f"HTTP {exc.code} from Moodle while requesting {safe_url}: {body}") from exc


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
        safe_url = re.sub(r"secret=[^&]+", "secret=<hidden>", url)
        raise RuntimeError(f"HTTP {exc.code} from Moodle while requesting {safe_url}: {body}") from exc


def fetch_moodle_bootstrap():
    query = urllib.parse.urlencode({
        "cmid": MOODLE_CMID,
        "secret": MOODLE_API_SECRET,
        "profile": MOODLE_ALGORITHM_PROFILE,
    })
    url = moodle_endpoint(f"station_bootstrap.php?{query}")
    safe_url = re.sub(r"secret=[^&]+", "secret=<hidden>", url)
    print(f"[MOODLE] GET {safe_url}")
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
    if HIDE_CONFIRMED_KNOWN_FACES:
        print(
            "Confirmed known faces hidden after "
            f"{CONFIRMED_KNOWN_STABLE_FRAMES} stable frame(s) and similarity >= "
            f"{CONFIRMED_KNOWN_SIMILARITY_THRESHOLD:.2f}"
        )
    if HIDE_CONFIRMED_UNKNOWN_FACES:
        print(f"Repeated unknown faces hidden after {CONFIRMED_UNKNOWN_STABLE_FRAMES} detection(s).")
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
        "confirmed_hidden": False,
        "confirmed_similarity": -1.0,
        "confirmed_at": 0.0,
        "recognition_count": 0,
        "recognition_stopped": False,
        "stopped_at": 0.0,
    })
    unknown_state_by_id = {"_next_number": 1}
    identity_state_by_track = {"_next_number": 1}

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
            resolved_skip_targets = build_resolved_face_skip_targets(face_state_by_name, unknown_state_by_id)
            candidates = detect_faces_with_search_zoom(
                detector=detector,
                recognizer=recognizer,
                known_features=known_features,
                frame=frame,
                full_grid=should_full_grid_search,
                last_box=None,
                locked_name=None,
                resolved_skip_targets=resolved_skip_targets,
                frame_number=frame_index,
            )

            # Do NOT filter by the entrance zone here. The goal is maximum faces.
            # Multi-scale/grid crops can detect the same physical face several times,
            # so dedupe after collecting all detections.
            candidates = dedupe_candidates(candidates)
            candidates = suppress_unknown_near_known(candidates)
            update_identity_vote_tracking(candidates, identity_state_by_track, now)
            last_candidates = candidates

        # Full-frame display: do not zoom into one face, because that hides other people.
        display_frame = frame.copy()

        # Update stability and attendance for every known face independently.
        seen_known_names = set()
        for candidate in candidates:
            if is_candidate_pending_identity(candidate):
                continue
            if not is_candidate_accepted_known(candidate):
                continue

            name = candidate["name"]

            # If the same person is found twice after dedupe, keep only the best one for state.
            if name in seen_known_names:
                continue
            seen_known_names.add(name)

            state = face_state_by_name[name]
            state["last_seen"] = now
            state["last_box"] = candidate["box"]
            state["last_similarity"] = candidate.get("similarity", -1.0)

            # After the configured recognition limit is reached, keep only the
            # lightweight box tracking alive. Do not keep increasing counters,
            # saving proof photos, or marking attendance for that person.
            if state.get("recognition_stopped", False):
                state["confirmed_hidden"] = True
                continue

            state["stable_count"] += 1
            state["recognition_count"] = int(state.get("recognition_count", 0) or 0) + 1
            mark_candidate_confirmed_if_ready(name, candidate, state, now)
            mark_known_stopped_if_ready(name, candidate, state, now)

            if not state.get("recognition_stopped", False) and state["stable_count"] >= STABLE_FRAMES_REQUIRED and SAVE_PHOTO_WHEN_KNOWN_STABLE:
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

        review_unknown_candidates = [
            c for c in candidates
            if not c.get("is_known", False) and not is_candidate_pending_identity(c)
        ]
        update_unknown_tracking(review_unknown_candidates, unknown_state_by_id, now)

        # Send unknown faces to Moodle for teacher review, but throttle to avoid flooding the server.
        if USE_MOODLE_API and current_moodle_session_id() and (now - last_unknown_moodle_save_time >= UNKNOWN_MOODLE_COOLDOWN_SECONDS):
            for candidate in candidates:
                if candidate.get("is_known", False):
                    continue
                if is_candidate_confirmed_unknown_hidden(candidate, unknown_state_by_id):
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
        cleanup_unknown_tracking(unknown_state_by_id, now)

        # Draw only unresolved faces directly on the original full frame.
        # Confirmed known faces stay active internally, but their boxes are hidden
        # so the operator can focus on people who still need recognition/review.
        visible_candidates, hidden_confirmed_candidates = split_candidates_by_confirmation(candidates, face_state_by_name, unknown_state_by_id)

        unknown_draw_index = 0
        for candidate in visible_candidates:
            is_known = is_candidate_accepted_known(candidate)
            is_pending_identity = is_candidate_pending_identity(candidate)

            if is_known and not DRAW_ALL_RECOGNIZED_FACES:
                continue
            if not candidate.get("is_known", False) and not DRAW_UNKNOWN_FACES:
                continue

            x1, y1, x2, y2 = candidate["box"]
            x1 = max(0, min(frame_w - 1, int(x1)))
            y1 = max(0, min(frame_h - 1, int(y1)))
            x2 = max(0, min(frame_w - 1, int(x2)))
            y2 = max(0, min(frame_h - 1, int(y2)))

            name = candidate.get("name", UNKNOWN_LABEL)
            color = (0, 255, 0) if is_known else ((0, 170, 255) if is_pending_identity else (0, 0, 255))

            if is_known:
                label = short_display_name(display_name_for_person(name))
                if DRAW_SIMILARITY_ON_LABEL and "similarity" in candidate:
                    label += f" {candidate['similarity']:.2f}"
                if DRAW_STABILITY_ON_LABEL:
                    stable_count = face_state_by_name.get(name, {}).get("stable_count", 0)
                    label += f" {min(stable_count, STABLE_FRAMES_REQUIRED)}/{STABLE_FRAMES_REQUIRED}"
            elif is_pending_identity:
                raw_name = str(candidate.get("raw_name") or candidate.get("name") or "")
                label = "CHECK " + short_display_name(display_name_for_person(raw_name))
                votes = int(candidate.get("identity_vote_count", 0) or 0)
                label += f" {min(votes, IDENTITY_MIN_KNOWN_VOTES)}/{IDENTITY_MIN_KNOWN_VOTES}"
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

        visible_known_count = sum(1 for c in visible_candidates if is_candidate_accepted_known(c))
        visible_pending_count = sum(1 for c in visible_candidates if is_candidate_pending_identity(c))
        visible_unknown_count = len(visible_candidates) - visible_known_count - visible_pending_count
        hidden_confirmed_count = len(hidden_confirmed_candidates)
        status = (
            f"Detected {len(candidates)} face(s) | Remaining {len(visible_candidates)}"
            f" | Known {visible_known_count} | Unknown {visible_unknown_count}"
        )
        if visible_pending_count:
            status += f" | Checking {visible_pending_count}"
        if hidden_confirmed_count:
            status += f" | Resolved hidden {hidden_confirmed_count}"
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
            unknown_state_by_id.clear()
            unknown_state_by_id["_next_number"] = 1
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
