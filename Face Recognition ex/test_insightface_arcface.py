import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ============================================================
# InsightFace / ArcFace test program
# ============================================================
#
# What this does:
# 1. Loads known people from an images folder.
# 2. Extracts ArcFace embeddings using InsightFace.
# 3. Opens the webcam.
# 4. Detects faces live.
# 5. Compares live embeddings with known embeddings using cosine similarity.
# 6. Draws green label for known people and red label for Unknown.
#
# Recommended folder structure:
#
# images/
#   Lucian/
#     1.jpg
#     2.jpg
#     3.jpg
#   John/
#     1.jpg
#     2.jpg
#
# Flat structure also works:
#
# images/
#   Lucian_1.jpg
#   Lucian_2.jpg
#   John_1.jpg
#
# Install:
#
# pip install insightface onnxruntime opencv-python numpy
#
# If you have NVIDIA GPU and CUDA configured:
#
# pip install insightface onnxruntime-gpu opencv-python numpy
#
# Run:
#
# python test_insightface_arcface.py
#
# If your camera is index 0:
#
# python test_insightface_arcface.py --camera 0
#
# If recognition is too strict:
#
# python test_insightface_arcface.py --threshold 0.35
#
# If recognition is too permissive/wrong:
#
# python test_insightface_arcface.py --threshold 0.50
# ============================================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNKNOWN_LABEL = "Unknown"


def parse_args():
    parser = argparse.ArgumentParser(description="Test InsightFace ArcFace recognition from webcam.")

    parser.add_argument(
        "--images",
        default="images",
        help="Folder with known face images. Default: images",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Camera index. Default: 1. Try 0 if camera does not open.",
    )

    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Detection input size. Higher can detect smaller/farther faces but is slower. Default: 640",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.42,
        help=(
            "Cosine distance threshold. Lower = stricter, higher = more permissive. "
            "Good starting range: 0.35-0.50. Default: 0.42"
        ),
    )

    parser.add_argument(
        "--model-pack",
        default="buffalo_l",
        help=(
            "InsightFace model pack. buffalo_l is accurate but heavier. "
            "buffalo_s is faster/lighter. Default: buffalo_l"
        ),
    )

    parser.add_argument(
        "--camera-width",
        type=int,
        default=1280,
        help="Requested camera width. Default: 1280",
    )

    parser.add_argument(
        "--camera-height",
        type=int,
        default=720,
        help="Requested camera height. Default: 720",
    )

    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save failed/unknown face crops into debug_unknown/.",
    )

    return parser.parse_args()


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Both vectors should already be L2-normalized.
    Cosine distance = 1 - cosine similarity.
    Lower distance means better match.
    """
    return 1.0 - float(np.dot(a, b))


def get_person_name_from_path(image_path: Path, images_root: Path) -> str:
    """
    Supports two enrollment styles:

    1. Subfolder style:
       images/Lucian/1.jpg -> Lucian

    2. Flat-file style:
       images/Lucian_1.jpg -> Lucian
       images/Lucian.jpg   -> Lucian
    """
    relative = image_path.relative_to(images_root)

    if len(relative.parts) >= 2:
        return relative.parts[0]

    stem = image_path.stem

    if "_" in stem:
        return stem.split("_")[0]

    return stem


def find_image_files(images_dir: Path) -> List[Path]:
    files = []

    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)

    return sorted(files)


def load_known_embeddings(app, images_dir: str) -> Dict[str, List[np.ndarray]]:
    images_root = Path(images_dir)

    if not images_root.exists():
        raise RuntimeError(f"Images folder does not exist: {images_root.resolve()}")

    image_files = find_image_files(images_root)

    if not image_files:
        raise RuntimeError(f"No image files found in: {images_root.resolve()}")

    known: Dict[str, List[np.ndarray]] = {}

    print(f"Loading known faces from: {images_root.resolve()}")
    print(f"Found {len(image_files)} image files.")

    for image_path in image_files:
        person_name = get_person_name_from_path(image_path, images_root)

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] Could not read image: {image_path}")
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"[SKIP] No face detected in: {image_path}")
            continue

        if len(faces) > 1:
            print(f"[WARN] Multiple faces in {image_path}; using largest face.")

        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        embedding = l2_normalize(largest_face.embedding.astype(np.float32))

        known.setdefault(person_name, []).append(embedding)

        print(f"[OK] {person_name}: {image_path.name}")

    if not known:
        raise RuntimeError("No valid face embeddings were loaded.")

    print("\nLoaded people:")
    for name, embeddings in known.items():
        print(f"  {name}: {len(embeddings)} embeddings")

    return known


def recognize_face(
    live_embedding: np.ndarray,
    known_embeddings: Dict[str, List[np.ndarray]],
    threshold: float,
) -> Tuple[str, float]:
    """
    Compare live embedding against all known embeddings.

    Returns:
      name, best_distance

    If best distance is above threshold, returns Unknown.
    """
    live_embedding = l2_normalize(live_embedding.astype(np.float32))

    best_name = UNKNOWN_LABEL
    best_distance = 999.0

    for person_name, embeddings in known_embeddings.items():
        for known_embedding in embeddings:
            distance = cosine_distance(live_embedding, known_embedding)

            if distance < best_distance:
                best_distance = distance
                best_name = person_name

    if best_distance > threshold:
        return UNKNOWN_LABEL, best_distance

    return best_name, best_distance


def draw_face(frame, face, name: str, distance: float):
    x1, y1, x2, y2 = face.bbox.astype(int)

    color = (0, 255, 0) if name != UNKNOWN_LABEL else (0, 0, 255)

    label = f"{name} {distance:.3f}" if name != UNKNOWN_LABEL else f"Unknown {distance:.3f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label_y = max(30, y1 - 10)

    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        color,
        2,
    )


def save_unknown_crop(frame, face, index: int):
    out_dir = Path("debug_unknown")
    out_dir.mkdir(exist_ok=True)

    x1, y1, x2, y2 = face.bbox.astype(int)

    h, w = frame.shape[:2]

    margin_x = int((x2 - x1) * 0.35)
    margin_y = int((y2 - y1) * 0.35)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    crop = frame[y1:y2, x1:x2]

    if crop.size > 0:
        path = out_dir / f"unknown_{index}.jpg"
        cv2.imwrite(str(path), crop)


def open_camera(camera_index: int, width: int, height: int):
    # CAP_DSHOW helps on Windows. If it fails, fallback to default backend.
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def main():
    args = parse_args()

    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: insightface. Install with:\n"
            "pip install insightface onnxruntime opencv-python numpy"
        ) from exc

    providers = ["CPUExecutionProvider"]

    # ctx_id=-1 means CPU.
    # If using GPU with onnxruntime-gpu, you can change ctx_id to 0.
    app = FaceAnalysis(
        name=args.model_pack,
        providers=providers,
    )

    app.prepare(
        ctx_id=-1,
        det_size=(args.det_size, args.det_size),
    )

    known_embeddings = load_known_embeddings(app, args.images)

    cap = open_camera(args.camera, args.camera_width, args.camera_height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try --camera 0 or --camera 1.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nCamera resolution: {actual_w}x{actual_h}")
    print("Press ESC to exit.")
    print("Press S to save unknown face crops for debugging.\n")

    unknown_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        faces = app.get(frame)

        # Largest faces first; usually useful for attendance.
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        for face in faces:
            name, distance = recognize_face(face.embedding, known_embeddings, args.threshold)
            draw_face(frame, face, name, distance)

        status = (
            f"Model: {args.model_pack} | threshold: {args.threshold:.2f} | "
            f"faces: {len(faces)}"
        )

        cv2.putText(
            frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("InsightFace ArcFace Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key in (ord("s"), ord("S")) or args.save_debug:
            for face in faces:
                name, distance = recognize_face(face.embedding, known_embeddings, args.threshold)

                if name == UNKNOWN_LABEL:
                    unknown_index += 1
                    save_unknown_crop(frame, face, unknown_index)
                    print(f"Saved unknown crop #{unknown_index}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
