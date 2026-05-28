"""
Variant 01: OpenCV YuNet detector + OpenCV SFace recognizer.

This is the closest model family to your current script, but this version builds
the embeddings directly from a folder of subject images before opening the camera.

Install CPU:
  py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
  py -m pip install opencv-contrib-python numpy

Run CPU/auto:
  py variant_01_opencv_yunet_sface_from_images.py --subjects-dir images --device auto

Try CUDA only if your cv2 was built with OpenCV DNN CUDA support:
  py variant_01_opencv_yunet_sface_from_images.py --subjects-dir images --device cuda
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
    print_header,
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



def opencv_dnn_device(device: str) -> Tuple[int, int, str]:
    """
    Returns backend_id, target_id, effective_device_label.
    CUDA works only with a custom OpenCV build compiled with CUDA DNN support.
    The normal opencv-contrib-python pip wheel is usually CPU-only for this path.
    """
    requested = device.lower()

    if requested in {"auto", "cuda"}:
        has_cuda_constants = hasattr(cv2.dnn, "DNN_BACKEND_CUDA") and hasattr(cv2.dnn, "DNN_TARGET_CUDA")
        try:
            cuda_devices = cv2.cuda.getCudaEnabledDeviceCount() if hasattr(cv2, "cuda") else 0
        except Exception:
            cuda_devices = 0

        if has_cuda_constants and cuda_devices > 0:
            return cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "cuda"

        if requested == "cuda":
            print("[WARN] CUDA requested, but this OpenCV build does not expose a usable CUDA DNN device/backend. Using CPU.")
        return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "cpu"

    return cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU, "cpu"


def create_detector(score_threshold: float, nms_threshold: float, top_k: int, backend_id: int, target_id: int):
    try:
        return cv2.FaceDetectorYN_create(
            str(YUNET_MODEL),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
            backend_id,
            target_id,
        )
    except TypeError:
        if backend_id != cv2.dnn.DNN_BACKEND_OPENCV or target_id != cv2.dnn.DNN_TARGET_CPU:
            print("[WARN] Your OpenCV FaceDetectorYN API does not accept backend/target. Using CPU.")
        return cv2.FaceDetectorYN_create(
            str(YUNET_MODEL),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )
    except cv2.error as exc:
        print(f"[WARN] Could not create CUDA YuNet detector: {exc}. Using CPU.")
        return cv2.FaceDetectorYN_create(
            str(YUNET_MODEL),
            "",
            (320, 320),
            score_threshold,
            nms_threshold,
            top_k,
        )


def create_recognizer(backend_id: int, target_id: int):
    try:
        return cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "", backend_id, target_id)
    except TypeError:
        if backend_id != cv2.dnn.DNN_BACKEND_OPENCV or target_id != cv2.dnn.DNN_TARGET_CPU:
            print("[WARN] Your OpenCV FaceRecognizerSF API does not accept backend/target. Using CPU.")
        return cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")
    except cv2.error as exc:
        print(f"[WARN] Could not create CUDA SFace recognizer: {exc}. Using CPU.")
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto tries CUDA when available, otherwise CPU.")
    parser.add_argument("--gpu", action="store_true", help="Compatibility shortcut for --device cuda.")
    parser.add_argument("--window", default="Variant 01 - OpenCV YuNet + SFace")
    args = parser.parse_args()

    if args.gpu:
        args.device = "cuda"

    configure_opencv_threads()
    check_opencv_face_api()
    ensure_model(YUNET_MODEL, YUNET_URL)
    ensure_model(SFACE_MODEL, SFACE_URL)

    backend_id, target_id, effective_device = opencv_dnn_device(args.device)
    print_header("Variant 01 - OpenCV YuNet + SFace")
    print(f"Requested device: {args.device}")
    print(f"Effective device: {effective_device}")
    print(f"OpenCV version: {cv2.__version__}")

    detector = create_detector(args.det_threshold, args.nms_threshold, args.top_k, backend_id, target_id)
    recognizer = create_recognizer(backend_id, target_id)

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
            f"OpenCV YuNet+SFace [{effective_device}] | faces={len(faces)} | fps={current_fps:.1f} | ESC exit",
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
