"""
TensorFlow / FaceNet variant of the classroom multi-face attendance prototype.

This version replaces:
  OpenCV YuNet detector + OpenCV SFace recognizer

with:
  MTCNN face detector + TensorFlow/Keras FaceNet embeddings.

Why this is a separate file:
  - SFace embeddings and FaceNet embeddings are NOT compatible.
  - This script builds a new FaceNet embedding database from a folder of subject photos.

Recommended folder structure:
  subject_images/
      GhinescuLucian/
          1.jpg
          2.jpg
          front.jpg
      PopescuAna/
          1.jpg
          2.jpg

Alternative flat structure:
  subject_images/
      GhinescuLucian_1.jpg
      GhinescuLucian_2.jpg
      PopescuAna_1.jpg

Install examples:

  CPU / simple test:
    py -m pip install opencv-python numpy tensorflow mtcnn keras-facenet

  Windows native NVIDIA GPU:
    TensorFlow native Windows GPU support is historically tied to TF 2.10.x.
    For modern TensorFlow GPU on Windows, WSL2 is usually the cleaner route.

  WSL2 / Linux GPU:
    python -m pip install "tensorflow[and-cuda]" mtcnn keras-facenet opencv-python numpy

Run:
  py main_tensorflow_facenet_many_faces_attendance.py

Notes:
  - First run may take longer because the FaceNet package may initialize/download weights.
  - The first run also builds tf_facenet_embeddings_cache.json from subject_images/.
  - If you add/change subject images, set REBUILD_EMBEDDINGS_CACHE = True or delete the cache.
"""

import os

# Reduce TensorFlow log noise before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import tensorflow as tf
except Exception as exc:
    raise RuntimeError(
        "TensorFlow is not installed or failed to import.\n"
        "Install with:\n"
        "  py -m pip install tensorflow mtcnn keras-facenet opencv-python numpy\n"
        f"\nOriginal error: {exc}"
    ) from exc

try:
    from mtcnn.mtcnn import MTCNN
except Exception as exc:
    raise RuntimeError(
        "The mtcnn package is not installed or failed to import.\n"
        "Install with:\n"
        "  py -m pip install mtcnn\n"
        f"\nOriginal error: {exc}"
    ) from exc

try:
    from keras_facenet import FaceNet
except Exception as exc:
    raise RuntimeError(
        "The keras-facenet package is not installed or failed to import.\n"
        "Install with:\n"
        "  py -m pip install keras-facenet\n"
        f"\nOriginal error: {exc}"
    ) from exc


cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass


# ============================================================
# Configuration
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540
WINDOW_NAME = "TensorFlow FaceNet Many Faces"

# Folder with subject photos.
# Best structure:
#   subject_images/StudentName/1.jpg
#   subject_images/StudentName/2.jpg
SUBJECT_IMAGES_SOURCE = "subject_images"

# Cache generated FaceNet embeddings so startup is faster after the first run.
EMBEDDINGS_CACHE_PATH = "tf_facenet_embeddings_cache.json"
REBUILD_EMBEDDINGS_CACHE = False

CAPTURE_DIR = "captures_tf"
ATTENDANCE_CSV = "attendance_tf.csv"

UNKNOWN_LABEL = "Unknown"
UNKNOWN_SHORT_LABEL = "UNK"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# TensorFlow/GPU behavior.
USE_TENSORFLOW_GPU = True
TENSORFLOW_MEMORY_GROWTH = True

# MTCNN detector threshold.
MTCNN_CONFIDENCE_THRESHOLD = 0.90

# FaceNet embedding similarity threshold.
# FaceNet and SFace thresholds are different. Tune this using your real camera.
# If too many unknowns: decrease to 0.55.
# If wrong names appear: increase to 0.70+.
FACENET_SIMILARITY_THRESHOLD = 0.62

# FaceNet normally uses 160x160 RGB crops.
FACENET_INPUT_SIZE = 160

# Detection scaling.
# Lower = faster but weaker far/small face detection.
DETECTION_INPUT_WIDTH = 640

# Run detector every N frames.
# MTCNN is usually much slower than YuNet, so 2 or 3 may be better on CPU.
DETECT_EVERY_N_FRAMES = 1

# Optional multi-scale search. This is slower, but can help far faces.
ENABLE_PERIODIC_GRID_SEARCH = False
FULL_GRID_SEARCH_EVERY_N_FRAMES = 45
SEARCH_ZOOM_LEVELS_FAST = [1.0]
SEARCH_ZOOM_LEVELS_FULL = [1.0, 1.6]

# Tracking/attendance.
STABLE_FRAMES_REQUIRED = 5
FACE_STATE_TIMEOUT_SECONDS = 2.0
PHOTO_COOLDOWN_SECONDS = 8
ATTENDANCE_COOLDOWN_SECONDS = 60
MAX_PHOTOS_PER_PERSON = 1
SAVE_PHOTO_WHEN_KNOWN_STABLE = True

# Display.
DRAW_UNKNOWN_FACES = True
DRAW_ALL_RECOGNIZED_FACES = True
SHORT_LABELS = True
SHORT_LABEL_MAX_CHARS = 10
DRAW_SIMILARITY_ON_LABEL = True
DRAW_STABILITY_ON_LABEL = True

# Candidate cleanup.
CANDIDATE_IOU_DEDUPE_THRESHOLD = 0.30
SUPPRESS_UNKNOWN_NEAR_KNOWN = True
UNKNOWN_SUPPRESSION_IOU = 0.10
UNKNOWN_SUPPRESSION_CENTER_RATIO = 0.75
SAME_FACE_CENTER_RATIO = 0.55

# Proof photo zoom.
MIN_ZOOM = 1.0
MAX_ZOOM = 3.5
FACE_TARGET_HEIGHT = 0.38

# Entrance/door mode. Kept from your previous architecture, disabled by default.
ENTRANCE_MODE = False
ENTRANCE_ZONE_X1 = 0.25
ENTRANCE_ZONE_Y1 = 0.08
ENTRANCE_ZONE_X2 = 0.75
ENTRANCE_ZONE_Y2 = 0.95
DRAW_ENTRANCE_ZONE = False
ENTRANCE_SORT_BY_BIGGEST_FACE = True

os.makedirs(CAPTURE_DIR, exist_ok=True)


# ============================================================
# TensorFlow / FaceNet engine
# ============================================================

def configure_tensorflow():
    print(f"TensorFlow version: {tf.__version__}")

    physical_gpus = tf.config.list_physical_devices("GPU")

    if not USE_TENSORFLOW_GPU:
        try:
            tf.config.set_visible_devices([], "GPU")
            print("[TF] GPU disabled by config. Running on CPU.")
        except Exception as exc:
            print(f"[TF] Could not disable GPU after initialization: {exc}")
        return []

    if not physical_gpus:
        print("[TF] No GPU visible to TensorFlow. Running on CPU.")
        return []

    if TENSORFLOW_MEMORY_GROWTH:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as exc:
                print(f"[TF] Could not set memory growth on {gpu}: {exc}")

    logical_gpus = tf.config.list_logical_devices("GPU")
    print(f"[TF] Physical GPUs: {physical_gpus}")
    print(f"[TF] Logical GPUs: {logical_gpus}")
    return physical_gpus


def l2_normalize(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm == 0:
        return feature
    return feature / norm


def cosine_similarity(a, b):
    return float(np.dot(a, b))


class TensorFlowFaceNetEngine:
    def __init__(self):
        configure_tensorflow()

        print("[TF] Loading MTCNN detector...")
        self.detector = MTCNN()

        print("[TF] Loading FaceNet embedder...")
        self.embedder = FaceNet()

        print("[TF] TensorFlow FaceNet engine ready.\n")

    def detect_faces_rgb(self, rgb_image):
        detections = self.detector.detect_faces(rgb_image)
        clean = []

        h, w = rgb_image.shape[:2]

        for det in detections:
            confidence = float(det.get("confidence", 0.0))
            if confidence < MTCNN_CONFIDENCE_THRESHOLD:
                continue

            x, y, bw, bh = det.get("box", [0, 0, 0, 0])
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w - 1, int(x + bw))
            y2 = min(h - 1, int(y + bh))

            if x2 <= x1 or y2 <= y1:
                continue

            keypoints = det.get("keypoints", {}) or {}

            clean.append({
                "box": (x1, y1, x2, y2),
                "confidence": confidence,
                "keypoints": keypoints,
            })

        return clean

    def detect_faces_bgr(self, bgr_image):
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return self.detect_faces_rgb(rgb)

    def embed_face_crops_rgb(self, face_crops_rgb: List[np.ndarray]) -> np.ndarray:
        if not face_crops_rgb:
            return np.empty((0, 512), dtype=np.float32)

        prepared = []
        for crop in face_crops_rgb:
            if crop is None or crop.size == 0:
                continue

            if crop.shape[0] != FACENET_INPUT_SIZE or crop.shape[1] != FACENET_INPUT_SIZE:
                crop = cv2.resize(crop, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE), interpolation=cv2.INTER_AREA)

            prepared.append(crop.astype(np.uint8))

        if not prepared:
            return np.empty((0, 512), dtype=np.float32)

        embeddings = self.embedder.embeddings(np.asarray(prepared))
        return np.asarray([l2_normalize(e) for e in embeddings], dtype=np.float32)


# ============================================================
# Geometry and preprocessing
# ============================================================

def box_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def face_size_xyxy(box):
    x1, y1, x2, y2 = box
    return max(1.0, float(x2 - x1)), max(1.0, float(y2 - y1))


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def center_distance_ratio(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    aw, ah = face_size_xyxy(box_a)
    bw, bh = face_size_xyxy(box_b)
    reference = max(aw, ah, bw, bh, 1.0)

    return distance / reference


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


def candidates_are_same_physical_face(candidate_a, candidate_b, iou_threshold=CANDIDATE_IOU_DEDUPE_THRESHOLD):
    box_a = candidate_a["box"]
    box_b = candidate_b["box"]

    if intersection_over_union(box_a, box_b) >= iou_threshold:
        return True

    return center_distance_ratio(box_a, box_b) <= SAME_FACE_CENTER_RATIO


def dedupe_candidates(candidates, iou_threshold=CANDIDATE_IOU_DEDUPE_THRESHOLD):
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


def expand_and_clip_box(box, frame_w, frame_h, margin_ratio=0.22):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    margin_x = int(round(bw * margin_ratio))
    margin_y = int(round(bh * margin_ratio))

    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(frame_w - 1, x2 + margin_x),
        min(frame_h - 1, y2 + margin_y),
    )


def rotate_image_around_center(image, angle_degrees, center):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, matrix


def transform_point(matrix, point):
    x, y = point
    px = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    py = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return px, py


def align_crop_face_rgb(rgb_image, detection):
    """
    Returns a 160x160 RGB crop for FaceNet.

    MTCNN gives eye keypoints, so we lightly rotate the image to make the eyes horizontal.
    If keypoints are unavailable, it falls back to a padded box crop.
    """
    h, w = rgb_image.shape[:2]
    x1, y1, x2, y2 = detection["box"]
    keypoints = detection.get("keypoints", {}) or {}

    left_eye = keypoints.get("left_eye")
    right_eye = keypoints.get("right_eye")

    working_image = rgb_image
    working_box = (x1, y1, x2, y2)

    if left_eye is not None and right_eye is not None:
        lx, ly = left_eye
        rx, ry = right_eye

        dx = rx - lx
        dy = ry - ly

        if abs(dx) > 1:
            angle = np.degrees(np.arctan2(dy, dx))
            center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

            # Rotate opposite of eye slope.
            working_image, matrix = rotate_image_around_center(rgb_image, angle, center)

            corners = [
                transform_point(matrix, (x1, y1)),
                transform_point(matrix, (x2, y1)),
                transform_point(matrix, (x2, y2)),
                transform_point(matrix, (x1, y2)),
            ]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]

            working_box = (
                int(max(0, min(xs))),
                int(max(0, min(ys))),
                int(min(w - 1, max(xs))),
                int(min(h - 1, max(ys))),
            )

    crop_box = expand_and_clip_box(working_box, w, h, margin_ratio=0.22)
    cx1, cy1, cx2, cy2 = crop_box

    if cx2 <= cx1 or cy2 <= cy1:
        return None

    crop = working_image[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return crop


def resize_for_detection_bgr(frame_bgr):
    h, w = frame_bgr.shape[:2]

    if w <= DETECTION_INPUT_WIDTH:
        return frame_bgr, 1.0, 1.0

    scale = DETECTION_INPUT_WIDTH / float(w)
    new_w = DETECTION_INPUT_WIDTH
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    scale_back_x = w / float(new_w)
    scale_back_y = h / float(new_h)

    return resized, scale_back_x, scale_back_y


def scale_detection_to_original(detection, scale_back_x, scale_back_y):
    x1, y1, x2, y2 = detection["box"]
    new_detection = dict(detection)
    new_detection["box"] = (
        int(round(x1 * scale_back_x)),
        int(round(y1 * scale_back_y)),
        int(round(x2 * scale_back_x)),
        int(round(y2 * scale_back_y)),
    )

    keypoints = {}
    for name, point in (detection.get("keypoints", {}) or {}).items():
        px, py = point
        keypoints[name] = (
            int(round(px * scale_back_x)),
            int(round(py * scale_back_y)),
        )
    new_detection["keypoints"] = keypoints

    return new_detection


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


def map_crop_box_to_original(box, left, top, base_to_original_x, base_to_original_y):
    x1, y1, x2, y2 = box
    return (
        int(left + x1 * base_to_original_x),
        int(top + y1 * base_to_original_y),
        int(left + x2 * base_to_original_x),
        int(top + y2 * base_to_original_y),
    )


def map_crop_keypoints_to_original(keypoints, left, top, base_to_original_x, base_to_original_y):
    mapped = {}
    for name, point in (keypoints or {}).items():
        px, py = point
        mapped[name] = (
            int(left + px * base_to_original_x),
            int(top + py * base_to_original_y),
        )
    return mapped


def get_entrance_zone(frame_w, frame_h):
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


def split_display_name(name):
    cleaned = str(name).replace("_", " ").replace("-", " ").strip()

    if " " in cleaned:
        return [part for part in cleaned.split() if part]

    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", cleaned)
    return parts or [cleaned]


def short_display_name(name):
    if not SHORT_LABELS:
        return str(name)

    if not name or str(name) == UNKNOWN_LABEL:
        return UNKNOWN_SHORT_LABEL

    parts = split_display_name(name)
    label = parts[-1] if len(parts) >= 2 else parts[0]

    if len(label) > SHORT_LABEL_MAX_CHARS:
        label = label[:SHORT_LABEL_MAX_CHARS]

    return label


# ============================================================
# Subject photo loading / embedding cache
# ============================================================

def clean_person_name_from_path(image_path: Path, images_root: Path):
    relative = image_path.relative_to(images_root)

    if len(relative.parts) >= 2:
        return relative.parts[0]

    name = image_path.stem.strip()
    name = re.sub(r"[\s_-]*\d+$", "", name).strip()

    # Handles names like GhinescuLucian_front or PopescuAna_left.
    name = re.sub(r"[\s_-]*(front|left|right|down|up|straight|center|centre)$", "", name, flags=re.IGNORECASE).strip()

    return name or image_path.stem.strip()


def find_image_files(images_root: Path):
    return sorted(
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def serialize_known_features(known_features: Dict[str, List[np.ndarray]], path: str):
    data = {
        name: [feature.astype(float).tolist() for feature in features]
        for name, features in known_features.items()
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": {
                    "family": "tensorflow",
                    "detector": "mtcnn",
                    "recognizer": "keras-facenet/facenet",
                    "embedding_normalized": True,
                },
                "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                "people": data,
            },
            f,
            indent=2,
        )


def load_known_features_cache(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = data.get("people", data)
    known_features: Dict[str, List[np.ndarray]] = {}

    for name, features in people.items():
        if not isinstance(features, list):
            continue

        for feature in features:
            arr = np.asarray(feature, dtype=np.float32).reshape(-1)
            if arr.size < 64 or not np.all(np.isfinite(arr)):
                continue
            known_features.setdefault(str(name), []).append(l2_normalize(arr))

    if not known_features:
        raise RuntimeError(f"No valid features found in cache: {path}")

    print(f"Loaded FaceNet embedding cache: {path}")
    for name, features in known_features.items():
        print(f"  {name}: {len(features)} embedding(s)")
    print()

    return known_features


def load_known_features_from_subject_images(engine: TensorFlowFaceNetEngine, images_dir: str):
    images_root = Path(images_dir)

    if not images_root.exists():
        raise RuntimeError(
            f"Subject images folder does not exist: {images_root.resolve()}\n"
            "Create it like:\n"
            "  subject_images/GhinescuLucian/1.jpg\n"
            "  subject_images/GhinescuLucian/2.jpg"
        )

    image_files = find_image_files(images_root)

    if not image_files:
        raise RuntimeError(f"No image files found in: {images_root.resolve()}")

    known_features: Dict[str, List[np.ndarray]] = {}

    print(f"Loading subject photos from: {images_root.resolve()}")
    print(f"Found {len(image_files)} image file(s).\n")

    for image_path in image_files:
        person_name = clean_person_name_from_path(image_path, images_root)

        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"[SKIP] Could not read image: {image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = engine.detect_faces_rgb(rgb)

        if not detections:
            print(f"[SKIP] No face detected in: {image_path}")
            continue

        # For enrollment photos, use the largest detected face.
        largest = max(detections, key=lambda d: box_area_xyxy(d["box"]))

        crop = align_crop_face_rgb(rgb, largest)
        if crop is None:
            print(f"[SKIP] Could not crop face in: {image_path}")
            continue

        embedding = engine.embed_face_crops_rgb([crop])
        if embedding.shape[0] == 0:
            print(f"[SKIP] Could not embed face in: {image_path}")
            continue

        known_features.setdefault(person_name, []).append(embedding[0])
        print(f"[OK] {image_path.name} -> {person_name}")

    if not known_features:
        raise RuntimeError("No valid FaceNet embeddings were generated from subject photos.")

    print("\nLoaded people:")
    for name, features in known_features.items():
        print(f"  {name}: {len(features)} embedding(s)")
    print()

    serialize_known_features(known_features, EMBEDDINGS_CACHE_PATH)
    print(f"Saved embedding cache: {EMBEDDINGS_CACHE_PATH}\n")

    return known_features


def load_or_build_known_features(engine: TensorFlowFaceNetEngine):
    if not REBUILD_EMBEDDINGS_CACHE and Path(EMBEDDINGS_CACHE_PATH).exists():
        return load_known_features_cache(EMBEDDINGS_CACHE_PATH)

    return load_known_features_from_subject_images(engine, SUBJECT_IMAGES_SOURCE)


# ============================================================
# Recognition
# ============================================================

def recognize_feature(feature, known_features):
    best_name = UNKNOWN_LABEL
    best_similarity = -1.0

    for person_name, features in known_features.items():
        for known_feature in features:
            similarity = cosine_similarity(feature, known_feature)

            if similarity > best_similarity:
                best_similarity = similarity
                best_name = person_name

    if best_similarity < FACENET_SIMILARITY_THRESHOLD:
        return UNKNOWN_LABEL, best_similarity

    return best_name, best_similarity


def detect_faces_with_search_zoom_tf(engine, known_features, frame_bgr, full_grid=False, last_box=None):
    frame_h, frame_w = frame_bgr.shape[:2]
    candidates = []

    zoom_levels = SEARCH_ZOOM_LEVELS_FULL if full_grid else SEARCH_ZOOM_LEVELS_FAST

    search_crops = make_search_crops(
        frame_w,
        frame_h,
        zoom_levels=zoom_levels,
        full_grid=full_grid,
        last_box=last_box,
    )

    original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    for left, top, crop_w, crop_h, search_zoom, source in search_crops:
        crop_bgr = frame_bgr[top:top + crop_h, left:left + crop_w]

        if crop_bgr.size == 0:
            continue

        if search_zoom == 1.0:
            base_bgr = crop_bgr
            base_to_original_x = 1.0
            base_to_original_y = 1.0
        else:
            base_bgr = cv2.resize(crop_bgr, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
            base_to_original_x = crop_w / float(frame_w)
            base_to_original_y = crop_h / float(frame_h)

        detection_input_bgr, det_to_base_x, det_to_base_y = resize_for_detection_bgr(base_bgr)
        detection_input_rgb = cv2.cvtColor(detection_input_bgr, cv2.COLOR_BGR2RGB)

        detections_small = engine.detect_faces_rgb(detection_input_rgb)

        if not detections_small:
            continue

        original_detections = []
        face_crops = []

        for det_small in detections_small:
            det_base = scale_detection_to_original(det_small, det_to_base_x, det_to_base_y)

            original_box = map_crop_box_to_original(
                det_base["box"],
                left=left,
                top=top,
                base_to_original_x=base_to_original_x,
                base_to_original_y=base_to_original_y,
            )

            original_keypoints = map_crop_keypoints_to_original(
                det_base.get("keypoints", {}),
                left=left,
                top=top,
                base_to_original_x=base_to_original_x,
                base_to_original_y=base_to_original_y,
            )

            x1, y1, x2, y2 = original_box
            original_box = (
                max(0, min(frame_w - 1, x1)),
                max(0, min(frame_h - 1, y1)),
                max(0, min(frame_w - 1, x2)),
                max(0, min(frame_h - 1, y2)),
            )

            if original_box[2] <= original_box[0] or original_box[3] <= original_box[1]:
                continue

            det_original = {
                "box": original_box,
                "confidence": det_small.get("confidence", 0.0),
                "keypoints": original_keypoints,
            }

            face_crop = align_crop_face_rgb(original_rgb, det_original)
            if face_crop is None:
                continue

            original_detections.append(det_original)
            face_crops.append(face_crop)

        embeddings = engine.embed_face_crops_rgb(face_crops)

        for det_original, feature in zip(original_detections, embeddings):
            name, similarity = recognize_feature(feature, known_features)

            x1, y1, x2, y2 = det_original["box"]
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            is_known = bool(name) and name != UNKNOWN_LABEL

            candidates.append({
                "box": det_original["box"],
                "name": name,
                "is_known": is_known,
                "similarity": similarity,
                "area": box_w * box_h,
                "source": source,
                "search_zoom": search_zoom,
                "detector_confidence": det_original.get("confidence", 0.0),
            })

    return candidates


def choose_target_face(candidates, frame_w, frame_h, locked_name=None, last_box=None):
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

    if ENTRANCE_MODE:
        entrance_zone = get_entrance_zone(frame_w, frame_h)
        candidates.sort(key=lambda c: (-c["area"], zone_center_distance(c["box"], entrance_zone)))
    else:
        candidates.sort(key=lambda c: (-c["area"], center_distance(c["box"], frame_center_box)))

    return candidates[0]


# ============================================================
# Attendance/photo helpers
# ============================================================

def make_zoomed_proof_frame(frame, face_box):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = face_box
    face_h = max(1, y2 - y1)

    proof_zoom = (FACE_TARGET_HEIGHT * frame_h) / face_h
    proof_zoom = max(MIN_ZOOM, min(MAX_ZOOM, proof_zoom))

    crop_params = make_crop_around_box(frame_w, frame_h, proof_zoom, face_box)
    left, top, crop_w, crop_h = crop_params

    crop = frame[top:top + crop_h, left:left + crop_w]
    return cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ============================================================
# Main
# ============================================================

def main():
    global DRAW_UNKNOWN_FACES, ENABLE_PERIODIC_GRID_SEARCH

    engine = TensorFlowFaceNetEngine()
    known_features = load_or_build_known_features(engine)

    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"MTCNN confidence threshold: {MTCNN_CONFIDENCE_THRESHOLD}")
    print(f"FaceNet similarity threshold: {FACENET_SIMILARITY_THRESHOLD}")
    print("Press ESC to exit. Press R to reset tracking state.")
    print("Press U to toggle Unknown drawing. Press G to toggle slower grid/zoom search.\n")

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

    fps_last_time = time.time()
    fps_frames = 0
    measured_fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        now = time.time()
        frame_index += 1
        fps_frames += 1

        if now - fps_last_time >= 1.0:
            measured_fps = fps_frames / (now - fps_last_time)
            fps_last_time = now
            fps_frames = 0

        frame_h, frame_w = frame.shape[:2]

        should_fast_search = frame_index % DETECT_EVERY_N_FRAMES == 0
        should_full_grid_search = ENABLE_PERIODIC_GRID_SEARCH and frame_index % FULL_GRID_SEARCH_EVERY_N_FRAMES == 0

        candidates = last_candidates

        if should_fast_search or should_full_grid_search:
            candidates = detect_faces_with_search_zoom_tf(
                engine=engine,
                known_features=known_features,
                frame_bgr=frame,
                full_grid=should_full_grid_search,
                last_box=None,
            )

            candidates = dedupe_candidates(candidates)
            candidates = suppress_unknown_near_known(candidates)
            last_candidates = candidates

        display_frame = frame.copy()

        seen_known_names = set()
        for candidate in candidates:
            if not candidate.get("is_known", False):
                continue

            name = candidate["name"]

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

        for name in list(face_state_by_name.keys()):
            if now - face_state_by_name[name]["last_seen"] > FACE_STATE_TIMEOUT_SECONDS:
                del face_state_by_name[name]

        if DRAW_ENTRANCE_ZONE or ENTRANCE_MODE:
            zx1, zy1, zx2, zy2 = get_entrance_zone(frame_w, frame_h)
            cv2.rectangle(display_frame, (zx1, zy1), (zx2, zy2), (255, 255, 0), 2)

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

        known_count = sum(1 for c in candidates if c.get("is_known", False))
        unknown_count = len(candidates) - known_count

        status = (
            f"FPS {measured_fps:.1f} | Detected {len(candidates)} | "
            f"Known {known_count} | Unknown {unknown_count}"
        )

        if not DRAW_UNKNOWN_FACES:
            status += " | Unknown hidden"

        if should_full_grid_search:
            status += " | grid search"
        elif should_fast_search:
            status += " | TF detect"
        else:
            status += " | cached"

        if not ENABLE_PERIODIC_GRID_SEARCH:
            status += " | grid OFF"

        cv2.putText(
            display_frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            "ESC exit | R reset | U unknown | G grid",
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
    main()
