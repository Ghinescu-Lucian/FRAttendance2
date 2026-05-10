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


# ============================================================
# FAST ATTENDANCE: OpenCV YuNet + SFace + auto-zoom/search flow
# ============================================================
# This program keeps the execution path/principle of main_fast_attendance.py:
#   1. Search for faces even when the person is far from the camera.
#   2. Use multi-scale digital crop search: full frame, center crops,
#      last-known face crop, and occasional full 3x3 grid crop.
#   3. Choose one target face.
#   4. Smoothly center/zoom the displayed camera crop on that face.
#   5. Wait for the same known person for STABLE_FRAMES_REQUIRED frames.
#   6. Save one proof photo.
#   7. Mark attendance in attendance.csv.
#   8. Keep the lock briefly, then release so the system can search for another student.
#
# But instead of SimpleFacerec/dlib, this version uses:
#   YuNet = cv2.FaceDetectorYN
#   SFace = cv2.FaceRecognizerSF
#
# Install:
#   py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
#   py -m pip install opencv-contrib-python numpy
#
# Run:
#   py main_yunet_sface_fast_attendance.py
# ============================================================


# =========================
# Configuration
# =========================
CAMERA_INDEX = 1
# Folder or file containing precomputed embeddings.
# You can point this to:
#   - one file:   "images/GhinescuLucian_embeddings_1.json"
#   - a folder:   "images/"
# The loader accepts files named like *_embeddings*, embeddings_*.json, or normal .json files.
EMBEDDINGS_SOURCE = "images/"

# Kept only if you still want the old image-loading function for debugging.
ENCODINGS_DIR = "images/"
CAPTURE_DIR = "captures"
WINDOW_NAME = "YuNet + SFace Fast Attendance"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

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
YUNET_SCORE_THRESHOLD = 0.75
YUNET_NMS_THRESHOLD = 0.30
YUNET_TOP_K = 5000

# SFace cosine similarity threshold.
# Lower = accepts more matches but more risk of wrong label.
# Higher = stricter but more Unknown.
SFACE_SIMILARITY_THRESHOLD = 0.38

# Display zoom after a face target is found.
MIN_ZOOM = 1.0
MAX_ZOOM = 3.5
FACE_TARGET_HEIGHT = 0.38

CENTER_SMOOTHING = 0.22
ZOOM_SMOOTHING = 0.16

# Recognition input size for zoom-search crops.
# Lower = faster, but weaker far-face detection.
SEARCH_INPUT_WIDTH = 960

# Search behavior.
FAST_SEARCH_EVERY_N_FRAMES = 2
FULL_GRID_SEARCH_EVERY_N_FRAMES = 12

# Search zooms.
SEARCH_ZOOM_LEVELS_FAST = [1.0, 2.4, 3.4]
SEARCH_ZOOM_LEVELS_FULL = [1.0, 1.8, 2.6, 3.4]

# Tracking/follow behavior.
STABLE_FRAMES_REQUIRED = 2
PHOTO_COOLDOWN_SECONDS = 8
MAX_LOST_FRAMES_AFTER_LOCK = 12

# Save policy.
SAVE_PHOTO_WHEN_KNOWN_STABLE = True
MAX_PHOTOS_PER_PERSON = 1

DRAW_DEBUG_SEARCH_SOURCE = False

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

            if locked_name and name == locked_name:
                return candidates

        if source == "full" and any(c["is_known"] for c in candidates):
            break

    return candidates


def choose_target_face(candidates, frame_w, frame_h, locked_name=None, last_box=None):
    if not candidates:
        return None

    frame_center_box = (frame_w // 2, frame_h // 2, frame_w // 2 + 1, frame_h // 2 + 1)

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
        known_faces.sort(
            key=lambda c: (
                -c["similarity"],
                -c["area"],
                center_distance(c["box"], frame_center_box),
            )
        )
        return known_faces[0]

    candidates.sort(key=lambda c: (-c["area"], center_distance(c["box"], frame_center_box)))

    return candidates[0]


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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ============================================================
# Main execution path: mirrors main_fast_attendance.py
# ============================================================

def main():
    check_opencv_api()
    ensure_model_file(YUNET_MODEL, YUNET_URL)
    ensure_model_file(SFACE_MODEL, SFACE_URL)

    detector = create_detector()
    recognizer = create_recognizer()

    known_features = load_known_features_from_embeddings(EMBEDDINGS_SOURCE)

    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"SFace similarity threshold: {SFACE_SIMILARITY_THRESHOLD}")
    print("Press ESC to exit. Press R to reset the current lock.\n")

    zoomer = FaceCenterZoomer()

    locked_name = None
    last_box = None
    lost_frames = 0

    stable_name = None
    stable_count = 0

    last_capture_time_by_name = defaultdict(lambda: 0.0)
    saved_count_by_name = defaultdict(int)
    last_attendance_time_by_name = defaultdict(lambda: 0.0)

    attendance_follow_until = 0.0
    frame_index = 0
    last_candidates = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        frame_index += 1
        frame_h, frame_w = frame.shape[:2]

        should_fast_search = frame_index % FAST_SEARCH_EVERY_N_FRAMES == 0
        should_full_grid_search = frame_index % FULL_GRID_SEARCH_EVERY_N_FRAMES == 0

        candidates = last_candidates

        if should_fast_search or should_full_grid_search:
            candidates = detect_faces_with_search_zoom(
                detector=detector,
                recognizer=recognizer,
                known_features=known_features,
                frame=frame,
                full_grid=should_full_grid_search,
                last_box=last_box,
                locked_name=locked_name,
            )

            last_candidates = candidates

        target_face = choose_target_face(
            candidates,
            frame_w,
            frame_h,
            locked_name=locked_name,
            last_box=last_box,
        )

        if target_face is not None:
            last_box = target_face["box"]
            lost_frames = 0
        else:
            lost_frames += 1

            if locked_name and last_box is not None and lost_frames <= MAX_LOST_FRAMES_AFTER_LOCK:
                target_face = {
                    "box": last_box,
                    "name": locked_name,
                    "is_known": True,
                    "similarity": 1.0,
                    "source": "last known",
                    "search_zoom": 0,
                }
            else:
                if lost_frames > MAX_LOST_FRAMES_AFTER_LOCK:
                    locked_name = None
                    last_box = None
                    last_candidates = []

                target_face = None

        face_box = target_face["box"] if target_face else None
        crop_params = zoomer.update(frame.shape, face_box=face_box)
        display_frame = FaceCenterZoomer.apply_zoom(frame, crop_params)

        if target_face is not None:
            name = target_face["name"]
            is_known = target_face["is_known"]

            if locked_name and time.time() > attendance_follow_until > 0:
                locked_name = None
                last_box = None
                last_candidates = []
                attendance_follow_until = 0.0

            if is_known:
                if name == stable_name:
                    stable_count += 1
                else:
                    stable_name = name
                    stable_count = 1
            else:
                stable_name = None
                stable_count = 0

            if is_known and stable_count >= STABLE_FRAMES_REQUIRED:
                locked_name = name

                if SAVE_PHOTO_WHEN_KNOWN_STABLE:
                    now = time.time()

                    attendance_cooldown_ok = (
                        now - last_attendance_time_by_name[name] >= ATTENDANCE_COOLDOWN_SECONDS
                    )
                    photo_limit_ok = saved_count_by_name[name] < MAX_PHOTOS_PER_PERSON
                    photo_cooldown_ok = now - last_capture_time_by_name[name] >= PHOTO_COOLDOWN_SECONDS

                    if attendance_cooldown_ok and photo_limit_ok and photo_cooldown_ok:
                        clean_zoomed = FaceCenterZoomer.apply_zoom(frame, crop_params)
                        image_path = save_person_photo(name, clean_zoomed)
                        timestamp = mark_attendance(name, image_path)

                        last_capture_time_by_name[name] = now
                        last_attendance_time_by_name[name] = now
                        saved_count_by_name[name] += 1

                        attendance_follow_until = now + STOP_FOLLOWING_AFTER_ATTENDANCE_SECONDS

                        print(f"ATTENDANCE MARKED: {name} at {timestamp}")
                        print(f"Saved proof photo: {image_path}")

            x1, y1, x2, y2 = transform_box_to_zoomed_view(face_box, crop_params, frame_w, frame_h)

            color = (0, 255, 0) if is_known else (0, 0, 255)
            label = name
            if "similarity" in target_face:
                label += f" {target_face['similarity']:.3f}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                display_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85,
                color,
                2,
            )

            if DRAW_DEBUG_SEARCH_SOURCE:
                cv2.putText(
                    display_frame,
                    f"source: {target_face.get('source', '')}",
                    (x1, min(frame_h - 15, y2 + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
        else:
            stable_name = None
            stable_count = 0

        status = f"Zoom: {zoomer.zoom:.2f}x"

        if locked_name:
            status += f" | ATTENDANCE LOCK {locked_name}"
        elif target_face is not None:
            status += " | Centering/searching face"
        else:
            status += " | Searching far face"

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
            "ESC = exit | R = reset lock",
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
            locked_name = None
            last_box = None
            lost_frames = 0
            stable_name = None
            stable_count = 0
            last_candidates = []
            print("Reset lock. Attendance/photo counters were kept.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
