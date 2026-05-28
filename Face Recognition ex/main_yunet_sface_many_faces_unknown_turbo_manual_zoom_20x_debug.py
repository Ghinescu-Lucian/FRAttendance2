import os
import re
import time
import csv
import json
import urllib.request
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
# Run:
#   py main_yunet_sface_many_faces_unknown_fast_short.py
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

# Display/proof zoom limits. Increase MAX_ZOOM if your camera is far from the door.
MIN_ZOOM = 1.0
MAX_ZOOM = 20.0
FACE_TARGET_HEIGHT = 0.38

CENTER_SMOOTHING = 0.22
ZOOM_SMOOTHING = 0.16

# Manual display zoom controller.
# This changes only what you SEE in the OpenCV window. Detection/recognition still
# runs on the original full camera frame, so zooming does not hide faces from the model.
# Keyboard zoom step: + and - change zoom by MANUAL_ZOOM_STEP each key press.
MANUAL_ZOOM_ENABLED_BY_DEFAULT = False
MANUAL_ZOOM_START = 1.0
MANUAL_ZOOM_STEP = 0.25
MANUAL_PAN_STEP_RATIO = 0.08
ENABLE_MANUAL_ZOOM_TRACKBARS = True

# Recognition input size for zoom-search crops.
# Lower = faster, but weaker far-face detection.
# 512 is a better real-time compromise than 640 on normal laptops.
SEARCH_INPUT_WIDTH = 512

# ============================================================
# Turbo performance settings
# ============================================================
# Main speed rule: detect often, but do NOT run SFace recognition on every
# face in every frame. Recognized identities are reused while the face is
# tracked in approximately the same position.
ENABLE_TRACKED_IDENTITY_CACHE = True

# Re-check a known face every N frames. Higher = faster. Lower = safer.
RECOGNITION_REFRESH_FRAMES = 8

# Unknown faces are retried less often. This prevents the same unknown face
# from running SFace 30 times/second.
UNKNOWN_RETRY_FRAMES = 12

# How many frames a cached face can disappear before its identity is forgotten.
TRACK_MAX_AGE_FRAMES = 10

# Geometry matching thresholds for reusing identity between frames.
TRACK_MATCH_IOU_THRESHOLD = 0.12
TRACK_MATCH_CENTER_RATIO = 0.85

# Ignore SFace extraction for very tiny faces unless there is already a cache.
# This saves CPU on tiny far-away false positives. Reduce this to 14 if you
# want more aggressive far-face recognition.
MIN_FACE_HEIGHT_FOR_RECOGNITION = 18

# Hard cap on heavy SFace runs per frame. Detection can still draw more faces.
MAX_RECOGNITIONS_PER_FRAME = 6

# Show FPS/performance info in the console periodically.
PRINT_PERFORMANCE_EVERY_SECONDS = 3.0

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



class ManualDisplayZoomController:
    """
    Manual zoom/pan for the displayed camera view.

    Important: this controller does not crop the frame used by detection.
    It only crops/resizes the final frame shown in the OpenCV window and maps
    face boxes into that displayed crop.
    """

    def __init__(self):
        self.enabled = bool(MANUAL_ZOOM_ENABLED_BY_DEFAULT)
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, float(MANUAL_ZOOM_START)))
        self.center_x_ratio = 0.50
        self.center_y_ratio = 0.50
        self.window_name = None
        self.trackbars_created = False

    @staticmethod
    def _noop(_value):
        pass

    def setup_trackbars(self, window_name):
        if not ENABLE_MANUAL_ZOOM_TRACKBARS:
            return

        self.window_name = window_name

        try:
            # Attach the controls to the camera window so the user can change zoom live.
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.createTrackbar("Manual 0/1", window_name, int(self.enabled), 1, self._noop)
            cv2.createTrackbar("Zoom x100", window_name, int(round(self.zoom * 100)), int(round(MAX_ZOOM * 100)), self._noop)
            cv2.createTrackbar("Pan X %", window_name, int(round(self.center_x_ratio * 100)), 100, self._noop)
            cv2.createTrackbar("Pan Y %", window_name, int(round(self.center_y_ratio * 100)), 100, self._noop)
            self.trackbars_created = True
        except Exception as exc:
            self.trackbars_created = False
            print(f"[WARN] Could not create manual zoom trackbars: {exc}")
            print("[WARN] Keyboard manual zoom still works: M, +, -, W, A, S, D, C.")

    def _set_trackbar(self, name, value):
        if not self.trackbars_created or not self.window_name:
            return

        try:
            cv2.setTrackbarPos(name, self.window_name, int(value))
        except Exception:
            pass

    def sync_trackbars(self):
        self._set_trackbar("Manual 0/1", 1 if self.enabled else 0)
        self._set_trackbar("Zoom x100", int(round(self.zoom * 100)))
        self._set_trackbar("Pan X %", int(round(self.center_x_ratio * 100)))
        self._set_trackbar("Pan Y %", int(round(self.center_y_ratio * 100)))

    def read_trackbars(self):
        if not self.trackbars_created or not self.window_name:
            return

        try:
            self.enabled = cv2.getTrackbarPos("Manual 0/1", self.window_name) == 1
            raw_zoom = cv2.getTrackbarPos("Zoom x100", self.window_name) / 100.0
            raw_x = cv2.getTrackbarPos("Pan X %", self.window_name) / 100.0
            raw_y = cv2.getTrackbarPos("Pan Y %", self.window_name) / 100.0
        except Exception:
            return

        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, raw_zoom))
        self.center_x_ratio = max(0.0, min(1.0, raw_x))
        self.center_y_ratio = max(0.0, min(1.0, raw_y))

        # If the user drags Zoom below MIN_ZOOM, put the knob back to the valid range.
        if raw_zoom != self.zoom:
            self._set_trackbar("Zoom x100", int(round(self.zoom * 100)))

    def toggle_enabled(self):
        self.enabled = not self.enabled
        self.sync_trackbars()
        print(f"Manual display zoom: {'ON' if self.enabled else 'OFF'}")

    def center(self):
        self.center_x_ratio = 0.50
        self.center_y_ratio = 0.50
        self.sync_trackbars()
        print("Manual zoom centered.")

    def adjust_zoom(self, delta):
        self.enabled = True
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom + float(delta)))
        self.sync_trackbars()
        print(f"Manual display zoom: {self.zoom:.2f}x")

    def pan(self, dx_ratio, dy_ratio):
        self.enabled = True
        # Pan less when zoomed in so the movement remains controllable.
        zoom_adjusted_x = dx_ratio / max(self.zoom, 1.0)
        zoom_adjusted_y = dy_ratio / max(self.zoom, 1.0)
        self.center_x_ratio = max(0.0, min(1.0, self.center_x_ratio + zoom_adjusted_x))
        self.center_y_ratio = max(0.0, min(1.0, self.center_y_ratio + zoom_adjusted_y))
        self.sync_trackbars()

    def get_crop_params(self, frame_w, frame_h):
        if not self.enabled or self.zoom <= MIN_ZOOM + 0.001:
            return 0, 0, frame_w, frame_h

        zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))
        crop_w = max(1, int(round(frame_w / zoom)))
        crop_h = max(1, int(round(frame_h / zoom)))

        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        cx = self.center_x_ratio * frame_w
        cy = self.center_y_ratio * frame_h

        cx = max(half_w, min(frame_w - half_w, cx))
        cy = max(half_h, min(frame_h - half_h, cy))

        left = int(round(cx - half_w))
        top = int(round(cy - half_h))

        left = max(0, min(frame_w - crop_w, left))
        top = max(0, min(frame_h - crop_h, top))

        return left, top, crop_w, crop_h

    def apply(self, frame):
        frame_h, frame_w = frame.shape[:2]
        crop_params = self.get_crop_params(frame_w, frame_h)
        left, top, crop_w, crop_h = crop_params

        if crop_w == frame_w and crop_h == frame_h:
            return frame.copy(), crop_params

        crop = frame[top:top + crop_h, left:left + crop_w]
        display_frame = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        return display_frame, crop_params

    def describe(self):
        if not self.enabled:
            return "manual zoom OFF"
        return f"manual zoom {self.zoom:.2f}x | pan {int(self.center_x_ratio * 100)}%,{int(self.center_y_ratio * 100)}%"


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


def build_feature_gallery(known_features):
    """
    Flattens {person: [embedding, ...]} into one normalized NumPy matrix.
    This is much faster than nested Python loops during live recognition.
    """
    raw_items = []
    dimension_counts = defaultdict(int)

    for person_name, person_features in known_features.items():
        for feature in person_features:
            feature = np.asarray(feature, dtype=np.float32).reshape(-1)
            if feature.size == 0 or not np.all(np.isfinite(feature)):
                continue
            raw_items.append((person_name, feature))
            dimension_counts[int(feature.size)] += 1

    if not raw_items:
        raise RuntimeError("No valid gallery embeddings were available after normalization.")

    # Prefer SFace's 128-dim vectors when present; otherwise use the most common
    # dimension and skip inconsistent vectors instead of crashing np.vstack.
    if dimension_counts.get(128, 0) > 0:
        target_dim = 128
    else:
        target_dim = max(dimension_counts.items(), key=lambda item: item[1])[0]

    names = []
    features = []
    skipped = 0
    for person_name, feature in raw_items:
        if int(feature.size) != int(target_dim):
            skipped += 1
            continue
        names.append(person_name)
        features.append(l2_normalize(feature))

    if skipped:
        print(f"[WARN] Skipped {skipped} gallery embedding(s) with a non-{target_dim} dimension.")

    if not features:
        raise RuntimeError("No gallery embeddings matched the selected vector dimension.")

    gallery_matrix = np.vstack(features).astype(np.float32)
    # Safety: normalize the complete gallery one more time.
    norms = np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    gallery_matrix = gallery_matrix / norms

    gallery_names = np.asarray(names, dtype=object)
    return gallery_names, gallery_matrix


def recognize_feature_fast(feature, gallery_names, gallery_matrix):
    """Vectorized cosine search over all known embeddings."""
    if gallery_matrix is None or len(gallery_matrix) == 0:
        return UNKNOWN_LABEL, -1.0

    feature = l2_normalize(feature).astype(np.float32)

    # Because all vectors are L2-normalized, dot product = cosine similarity.
    similarities = gallery_matrix @ feature
    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])
    best_name = str(gallery_names[best_index])

    if best_similarity < SFACE_SIMILARITY_THRESHOLD:
        return UNKNOWN_LABEL, best_similarity

    return best_name, best_similarity


class FaceIdentityCache:
    """
    Lightweight IoU/center-distance tracking cache.

    It does not try to be a full tracker. It only answers one practical question:
    "Is this detected face in almost the same place as a face we already
    recognized recently?" If yes, reuse the label and skip the expensive SFace
    feature extraction for this frame.
    """

    def __init__(self):
        self.next_track_id = 1
        self.tracks = {}
        self.current_frame_index = 0
        self.recognitions_this_frame = 0
        self.cache_hits_this_frame = 0

    def clear(self):
        self.next_track_id = 1
        self.tracks.clear()
        self.recognitions_this_frame = 0
        self.cache_hits_this_frame = 0

    def begin_frame(self, frame_index):
        self.current_frame_index = int(frame_index)
        self.recognitions_this_frame = 0
        self.cache_hits_this_frame = 0
        self._prune_old_tracks()

    def _prune_old_tracks(self):
        old_ids = [
            track_id for track_id, track in self.tracks.items()
            if self.current_frame_index - int(track.get("last_seen_frame", -999999)) > TRACK_MAX_AGE_FRAMES
        ]
        for track_id in old_ids:
            del self.tracks[track_id]

    def _find_best_track(self, box):
        best_track_id = None
        best_track = None
        best_score = -1.0

        for track_id, track in self.tracks.items():
            age = self.current_frame_index - int(track.get("last_seen_frame", -999999))
            if age > TRACK_MAX_AGE_FRAMES:
                continue

            track_box = track.get("box")
            if track_box is None:
                continue

            iou = intersection_over_union(box, track_box)
            center_ratio = center_distance_ratio(box, track_box)

            same_face = (
                iou >= TRACK_MATCH_IOU_THRESHOLD or
                center_ratio <= TRACK_MATCH_CENTER_RATIO
            )
            if not same_face:
                continue

            score = iou + max(0.0, 1.0 - center_ratio)
            if score > best_score:
                best_score = score
                best_track_id = track_id
                best_track = track

        return best_track_id, best_track

    def _should_recognize(self, track, box):
        _x1, y1, _x2, y2 = box
        face_h = max(1, int(y2 - y1))

        if track is None:
            return face_h >= MIN_FACE_HEIGHT_FOR_RECOGNITION

        frames_since_recognition = self.current_frame_index - int(track.get("last_recognition_frame", -999999))
        cached_name = track.get("name", UNKNOWN_LABEL)
        is_known = cached_name != UNKNOWN_LABEL

        if is_known:
            return frames_since_recognition >= RECOGNITION_REFRESH_FRAMES

        # Unknown faces get retried, but not every frame.
        if face_h < MIN_FACE_HEIGHT_FOR_RECOGNITION:
            return False
        return frames_since_recognition >= UNKNOWN_RETRY_FRAMES

    def _recognition_budget_available(self):
        return self.recognitions_this_frame < MAX_RECOGNITIONS_PER_FRAME

    def assign_identity(self, candidate, recognition_input, face, recognizer, gallery_names, gallery_matrix):
        box = candidate["box"]
        track_id, track = self._find_best_track(box)

        should_recognize = (
            not ENABLE_TRACKED_IDENTITY_CACHE or
            self._should_recognize(track, box)
        )

        did_recognize = False
        if should_recognize and self._recognition_budget_available():
            feature = extract_feature(recognizer, recognition_input, face)
            name, similarity = recognize_feature_fast(feature, gallery_names, gallery_matrix)
            self.recognitions_this_frame += 1
            did_recognize = True
        elif track is not None:
            name = track.get("name", UNKNOWN_LABEL)
            similarity = float(track.get("similarity", -1.0))
            self.cache_hits_this_frame += 1
        else:
            name = UNKNOWN_LABEL
            similarity = -1.0

        is_known = bool(name) and name != UNKNOWN_LABEL

        if track_id is None:
            track_id = self.next_track_id
            self.next_track_id += 1

        last_recognition_frame = self.current_frame_index if did_recognize else (
            int(track.get("last_recognition_frame", -999999)) if track is not None else -999999
        )

        self.tracks[track_id] = {
            "box": box,
            "name": name,
            "is_known": is_known,
            "similarity": similarity,
            "last_seen_frame": self.current_frame_index,
            "last_recognition_frame": last_recognition_frame,
        }

        candidate.update({
            "track_id": track_id,
            "name": name,
            "is_known": is_known,
            "similarity": similarity,
            "identity_from_cache": not did_recognize,
        })
        return candidate


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


def detect_faces_with_search_zoom(detector, recognizer, gallery_names, gallery_matrix, frame, identity_cache=None, frame_index=0, full_grid=False, last_box=None, locked_name=None):
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

            candidate = {
                "box": original_box,
                "name": UNKNOWN_LABEL,
                "is_known": False,
                "similarity": -1.0,
                "area": area,
                "source": source,
                "search_zoom": search_zoom,
            }

            if identity_cache is not None:
                candidate = identity_cache.assign_identity(
                    candidate=candidate,
                    recognition_input=recognition_input,
                    face=face,
                    recognizer=recognizer,
                    gallery_names=gallery_names,
                    gallery_matrix=gallery_matrix,
                )
            else:
                feature = extract_feature(recognizer, recognition_input, face)
                name, similarity = recognize_feature_fast(feature, gallery_names, gallery_matrix)
                candidate["name"] = name
                candidate["similarity"] = similarity
                candidate["is_known"] = bool(name) and name != UNKNOWN_LABEL
                candidate["identity_from_cache"] = False

            candidates.append(candidate)

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


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    # MJPG often reduces USB camera latency/CPU compared with raw YUYV on Windows.
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
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
    gallery_names, gallery_matrix = build_feature_gallery(known_features)
    print(f"Optimized gallery matrix: {gallery_matrix.shape[0]} embedding(s) x {gallery_matrix.shape[1]} dimensions")

    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"MANUAL/DISPLAY MAX_ZOOM active in this running file: {MAX_ZOOM:.1f}x")
    print(f"YuNet score threshold: {YUNET_SCORE_THRESHOLD}")
    print(f"SFace similarity threshold: {SFACE_SIMILARITY_THRESHOLD}")
    print("Press ESC to exit. Press R to reset tracking state.")
    print("Turbo mode: full-frame detection + identity cache + vectorized gallery matching.")
    print(f"Recognition refresh: known every {RECOGNITION_REFRESH_FRAMES} frames, unknown every {UNKNOWN_RETRY_FRAMES} frames.")
    print(f"Max heavy SFace recognitions per frame: {MAX_RECOGNITIONS_PER_FRAME}")
    print("Manual display zoom: M toggle, + zoom in, - zoom out, W/A/S/D pan, C center.")
    print("Trackbars are also available at the top of the camera window.")
    print("Detection/recognition still runs on the original full camera frame while manual zoom is active.")
    print("Press U to toggle Unknown drawing. Press G to toggle slower grid/zoom search.\n")

    manual_zoom = ManualDisplayZoomController()
    manual_zoom.setup_trackbars(WINDOW_NAME)

    identity_cache = FaceIdentityCache()

    fps_last_time = time.time()
    fps_frame_count = 0
    fps_value = 0.0
    last_perf_print_time = time.time()
    perf_sface_runs = 0
    perf_cache_hits = 0

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

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        now = time.time()
        frame_index += 1
        frame_h, frame_w = frame.shape[:2]

        identity_cache.begin_frame(frame_index)

        should_fast_search = frame_index % FAST_SEARCH_EVERY_N_FRAMES == 0
        should_full_grid_search = ENABLE_PERIODIC_GRID_SEARCH and frame_index % FULL_GRID_SEARCH_EVERY_N_FRAMES == 0

        candidates = last_candidates

        if should_fast_search or should_full_grid_search:
            candidates = detect_faces_with_search_zoom(
                detector=detector,
                recognizer=recognizer,
                gallery_names=gallery_names,
                gallery_matrix=gallery_matrix,
                frame=frame,
                identity_cache=identity_cache,
                frame_index=frame_index,
                full_grid=should_full_grid_search,
                last_box=None,
                locked_name=None,
            )
            perf_sface_runs += identity_cache.recognitions_this_frame
            perf_cache_hits += identity_cache.cache_hits_this_frame

            # Do NOT filter by the entrance zone here. The goal is maximum faces.
            # Multi-scale/grid crops can detect the same physical face several times,
            # so dedupe after collecting all detections.
            candidates = dedupe_candidates(candidates)
            candidates = suppress_unknown_near_known(candidates)
            last_candidates = candidates

        # Display frame can be manually zoomed/panned, but detection is still performed
        # on the original full frame above.
        manual_zoom.read_trackbars()
        display_frame, display_crop_params = manual_zoom.apply(frame)

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

                    print(f"ATTENDANCE MARKED: {name} at {timestamp}")
                    print(f"Saved proof photo: {image_path}")

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

            original_box = candidate["box"]

            # Hide boxes that are outside the currently displayed manual zoom crop.
            # Boxes inside the crop are mapped from original-frame coordinates to display coordinates.
            if not box_intersects_crop(original_box, display_crop_params):
                continue

            x1, y1, x2, y2 = transform_box_to_zoomed_view(
                original_box,
                display_crop_params,
                frame_w,
                frame_h,
            )

            if x2 <= x1 + 2 or y2 <= y1 + 2:
                continue

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

        fps_frame_count += 1
        fps_now = time.time()
        elapsed_for_fps = fps_now - fps_last_time
        if elapsed_for_fps >= 1.0:
            fps_value = fps_frame_count / elapsed_for_fps
            fps_frame_count = 0
            fps_last_time = fps_now

        if fps_now - last_perf_print_time >= PRINT_PERFORMANCE_EVERY_SECONDS:
            print(
                f"PERF fps={fps_value:.1f} | SFace runs={perf_sface_runs} | "
                f"cache hits={perf_cache_hits} | active tracks={len(identity_cache.tracks)}"
            )
            perf_sface_runs = 0
            perf_cache_hits = 0
            last_perf_print_time = fps_now

        status += f" | FPS {fps_value:.1f} | SFace {identity_cache.recognitions_this_frame} | cache {identity_cache.cache_hits_this_frame}"
        status += f" | {manual_zoom.describe()}"

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
            "ESC exit | M manual | +/- zoom | WASD pan | C center",
            (20, frame_h - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            "R reset | U unknown | G grid",
            (20, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key in (ord("r"), ord("R")):
            face_state_by_name.clear()
            identity_cache.clear()
            last_candidates = []
            print("Reset tracking state and identity cache. Attendance/photo counters were kept.")

        if key in (ord("u"), ord("U")):
            DRAW_UNKNOWN_FACES = not DRAW_UNKNOWN_FACES
            state = "ON" if DRAW_UNKNOWN_FACES else "OFF"
            print(f"Unknown face drawing: {state}")

        if key in (ord("g"), ord("G")):
            ENABLE_PERIODIC_GRID_SEARCH = not ENABLE_PERIODIC_GRID_SEARCH
            state = "ON" if ENABLE_PERIODIC_GRID_SEARCH else "OFF"
            last_candidates = []
            print(f"Periodic grid/zoom search: {state}")

        if key in (ord("m"), ord("M")):
            manual_zoom.toggle_enabled()

        if key in (ord("c"), ord("C")):
            manual_zoom.center()

        if key in (ord("+"), ord("=")):
            manual_zoom.adjust_zoom(MANUAL_ZOOM_STEP)

        if key in (ord("-"), ord("_")):
            manual_zoom.adjust_zoom(-MANUAL_ZOOM_STEP)

        if key in (ord("w"), ord("W")):
            manual_zoom.pan(0.0, -MANUAL_PAN_STEP_RATIO)

        if key in (ord("s"), ord("S")):
            manual_zoom.pan(0.0, MANUAL_PAN_STEP_RATIO)

        if key in (ord("a"), ord("A")):
            manual_zoom.pan(-MANUAL_PAN_STEP_RATIO, 0.0)

        if key in (ord("d"), ord("D")):
            manual_zoom.pan(MANUAL_PAN_STEP_RATIO, 0.0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
