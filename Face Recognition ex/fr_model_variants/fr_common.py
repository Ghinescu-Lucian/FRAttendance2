"""
Shared helpers for the facial-recognition model test variants.

Expected subject image layouts:

  subjects/
    GhinescuLucian/
      front.jpg
      left.jpg
    IuliaSocarde/
      1.jpg
      2.jpg

Also accepted, but less clean:

  subjects/GhinescuLucian_1.jpg
  subjects/GhinescuLucian_2.jpg

The person name is taken from the first folder level when subfolders are used.
For flat files, trailing numeric suffixes are removed.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNKNOWN_LABEL = "Unknown"


def configure_opencv_threads() -> None:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(max(1, os.cpu_count() or 1))
    except Exception:
        pass


def find_image_files(images_root: Path) -> List[Path]:
    return sorted(
        p for p in images_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def clean_person_name_from_path(image_path: Path, images_root: Path) -> str:
    relative = image_path.relative_to(images_root)

    # Preferred layout: subjects/PersonName/image.jpg
    if len(relative.parts) >= 2:
        return relative.parts[0].strip()

    # Fallback layout: subjects/PersonName_1.jpg or PersonName1.jpg
    name = image_path.stem.strip()
    name = re.sub(r"[\s_-]*\d+$", "", name).strip()
    return name or image_path.stem.strip()


def l2_normalize(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(feature))
    if norm <= 0:
        return feature
    return feature / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def best_cosine_match(
    embedding: np.ndarray,
    gallery: Dict[str, List[np.ndarray]],
    threshold: float,
) -> Tuple[str, float]:
    best_name = UNKNOWN_LABEL
    best_score = -1.0

    embedding = l2_normalize(embedding)

    for name, embeddings in gallery.items():
        for known_embedding in embeddings:
            score = cosine_similarity(embedding, known_embedding)
            if score > best_score:
                best_score = score
                best_name = name

    if best_score < threshold:
        return UNKNOWN_LABEL, best_score

    return best_name, best_score


def best_l2_match(
    embedding: np.ndarray,
    gallery: Dict[str, List[np.ndarray]],
    threshold: float,
) -> Tuple[str, float]:
    """
    For dlib/face_recognition-style 128D descriptors.
    Lower distance is better. threshold is usually around 0.50-0.60.
    Returns a confidence-like score as 1 - distance.
    """
    best_name = UNKNOWN_LABEL
    best_distance = 999.0

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

    for name, embeddings in gallery.items():
        for known_embedding in embeddings:
            distance = float(np.linalg.norm(embedding - np.asarray(known_embedding, dtype=np.float32).reshape(-1)))
            if distance < best_distance:
                best_distance = distance
                best_name = name

    if best_distance > threshold:
        return UNKNOWN_LABEL, 1.0 - best_distance

    return best_name, 1.0 - best_distance


def short_label(name: str, max_chars: int = 12) -> str:
    if not name or name == UNKNOWN_LABEL:
        return "UNK"

    cleaned = str(name).replace("_", " ").replace("-", " ").strip()
    if " " in cleaned:
        parts = [p for p in cleaned.split() if p]
    else:
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", cleaned) or [cleaned]

    label = parts[-1] if len(parts) >= 2 else parts[0]
    return label[:max_chars]


def draw_face(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    name: str,
    score: float | None = None,
    score_prefix: str = "",
) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    is_known = name != UNKNOWN_LABEL
    color = (0, 255, 0) if is_known else (0, 0, 255)

    label = short_label(name)
    if score is not None:
        label += f" {score_prefix}{score:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.58,
        color,
        2,
    )


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class FpsCounter:
    def __init__(self) -> None:
        self.last = time.time()
        self.frames = 0
        self.fps = 0.0

    def update(self) -> float:
        self.frames += 1
        now = time.time()
        elapsed = now - self.last
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.last = now
        return self.fps


def print_gallery(gallery: Dict[str, List[np.ndarray]]) -> None:
    print("\nLoaded gallery:")
    for name, embeddings in sorted(gallery.items()):
        print(f"  {name}: {len(embeddings)} embedding(s)")
    print()
