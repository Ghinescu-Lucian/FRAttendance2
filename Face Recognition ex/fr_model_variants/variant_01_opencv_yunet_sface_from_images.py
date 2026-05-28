"""
Variant 01: OpenCV YuNet detector + OpenCV SFace recognizer.

This is the closest model family to your current script, but this version builds
the embeddings directly from a folder of subject images before opening the camera.

Install:
  py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
  py -m pip install opencv-contrib-python numpy

Run:
  py variant_01_opencv_yunet_sface_from_images.py --subjects-dir images
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from fr_common import (
    UNKNOWN_LABEL,
    best_cosine_match,
    clean_person_name_from_path,
    configure_opencv_threads,
    draw_face,
    find_image_files,
    l2_normalize,
    open_camera,
    print_gallery,
    FpsCounter,
)


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


def ensure_model(path: Path, url: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return

    print(f"Downloading {path.name}")
    urllib.request.urlretrieve(url, str(path))

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Could not download model: {path}")


def check_opencv_face_api() -> None:
    missing = []
    if not hasattr(cv2, "FaceDetectorYN_create"):
        missing.append("cv2.FaceDetectorYN_create")
    if not hasattr(cv2, "FaceRecognizerSF_create"):
        missing.append("cv2.FaceRecognizerSF_create")

    if missing:
        raise RuntimeError(
            "Your OpenCV build does not include YuNet/SFace APIs:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nInstall opencv-contrib-python, not only opencv-python."
        )


def create_detector(score_threshold: float, nms_threshold: float, top_k: int):
    return cv2.FaceDetectorYN_create(
        str(YUNET_MODEL),
        "",
        (320, 320),
        score_threshold,
        nms_threshold,
        top_k,
    )


def create_recognizer():
    return cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")


def detect_faces(detector, image: np.ndarray, score_threshold: float) -> np.ndarray:
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    detector.setScoreThreshold(score_threshold)
    _, faces = detector.detect(image)

    if faces is None:
        return np.empty((0, 15), dtype=np.float32)

    return faces


def face_to_xyxy(face: np.ndarray) -> Tuple[int, int, int, int]:
    x, y, w, h = face[:4]
    return int(x), int(y), int(x + w), int(y + h)


def face_area(face: np.ndarray) -> float:
    return max(1.0, float(face[2])) * max(1.0, float(face[3]))


def extract_embedding(recognizer, image: np.ndarray, face: np.ndarray) -> np.ndarray:
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return l2_normalize(feature)


def build_gallery(
    subjects_dir: Path,
    detector,
    recognizer,
    score_threshold: float,
) -> Dict[str, List[np.ndarray]]:
    if not subjects_dir.exists():
        raise RuntimeError(f"Subjects folder does not exist: {subjects_dir.resolve()}")

    image_files = find_image_files(subjects_dir)
    if not image_files:
        raise RuntimeError(f"No subject images found in: {subjects_dir.resolve()}")

    gallery: Dict[str, List[np.ndarray]] = {}

    print(f"Building SFace gallery from: {subjects_dir.resolve()}")
    print(f"Found {len(image_files)} image(s).\n")

    for image_path in image_files:
        person_name = clean_person_name_from_path(image_path, subjects_dir)
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] Could not read {image_path}")
            continue

        faces = detect_faces(detector, image, score_threshold)
        if len(faces) == 0:
            print(f"[SKIP] No face found: {image_path.name}")
            continue

        largest = max(faces, key=face_area)
        embedding = extract_embedding(recognizer, image, largest)
        gallery.setdefault(person_name, []).append(embedding)
        print(f"[OK] {image_path.name} -> {person_name}")

    if not gallery:
        raise RuntimeError("No valid SFace embeddings were created.")

    return gallery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects-dir", default="images", help="Folder containing known subject images.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--det-threshold", type=float, default=0.70)
    parser.add_argument("--nms-threshold", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--match-threshold", type=float, default=0.36, help="Cosine similarity threshold for SFace.")
    parser.add_argument("--window", default="Variant 01 - OpenCV YuNet + SFace")
    args = parser.parse_args()

    configure_opencv_threads()
    check_opencv_face_api()
    ensure_model(YUNET_MODEL, YUNET_URL)
    ensure_model(SFACE_MODEL, SFACE_URL)

    detector = create_detector(args.det_threshold, args.nms_threshold, args.top_k)
    recognizer = create_recognizer()

    gallery = build_gallery(Path(args.subjects_dir), detector, recognizer, args.det_threshold)
    print_gallery(gallery)

    cap = open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    fps = FpsCounter()
    print("Press ESC to exit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from camera.")
            break

        faces = detect_faces(detector, frame, args.det_threshold)

        for face in faces:
            embedding = extract_embedding(recognizer, frame, face)
            name, score = best_cosine_match(embedding, gallery, args.match_threshold)
            draw_face(frame, face_to_xyxy(face), name, score, score_prefix="cos=")

        current_fps = fps.update()
        cv2.putText(
            frame,
            f"OpenCV YuNet+SFace | faces={len(faces)} | fps={current_fps:.1f} | ESC exit",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        cv2.imshow(args.window, frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
