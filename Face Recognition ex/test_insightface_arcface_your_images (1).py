import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ============================================================
# InsightFace / ArcFace test program for your current images folder
# ============================================================
#
# This version is adapted for this style:
#
# images/
#   GhinescuLucian.jpg
#   IuliaSocarde.jpg
#   IuliaSocarde2.jpg
#   SergioCanu.jpg
#
# Naming behavior:
# - GhinescuLucian.jpg  -> GhinescuLucian
# - SergioCanu.jpg      -> SergioCanu
# - IuliaSocarde.jpg    -> IuliaSocarde
# - IuliaSocarde2.jpg   -> IuliaSocarde
#
# That means trailing numbers are treated as extra photos of the same person.
#
# Install:
#   pip install insightface onnxruntime opencv-python numpy
#
# Run:
#   python test_insightface_arcface_your_images.py
#
# If your webcam is index 0:
#   python test_insightface_arcface_your_images.py --camera 0
#
# If recognition is too strict:
#   python test_insightface_arcface_your_images.py --threshold 0.50
#
# If recognition gives wrong names too easily:
#   python test_insightface_arcface_your_images.py --threshold 0.35
# ============================================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNKNOWN_LABEL = "Unknown"


def parse_args():
    parser = argparse.ArgumentParser(
        description="InsightFace ArcFace recognition adapted for a flat images/ folder."
    )

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
        help=(
            "Detection input size. Higher can detect smaller/farther faces but is slower. "
            "Default: 640"
        ),
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
            "InsightFace model pack. buffalo_l is more accurate but heavier. "
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
        "--no-strip-trailing-numbers",
        action="store_true",
        help=(
            "Do not group files like Person1.jpg and Person2.jpg as Person. "
            "By default trailing numbers are stripped."
        ),
    )

    return parser.parse_args()


def clean_person_name_from_filename(image_path: Path, strip_trailing_numbers: bool = True) -> str:
    """
    Converts a filename into a person label.

    Examples:
      GhinescuLucian.jpg -> GhinescuLucian
      IuliaSocarde2.jpg  -> IuliaSocarde
      Sergio_Canu_2.jpg  -> Sergio_Canu

    The trailing-number stripping helps when you have multiple images of the same person.
    """
    name = image_path.stem.strip()

    if strip_trailing_numbers:
        # Remove final digits and optional separator before them:
        # IuliaSocarde2 -> IuliaSocarde
        # IuliaSocarde_2 -> IuliaSocarde
        # IuliaSocarde-2 -> IuliaSocarde
        name = re.sub(r"[\s_-]*\d+$", "", name).strip()

    return name


def find_image_files(images_dir: Path) -> List[Path]:
    files = []

    for path in images_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(path)

    return sorted(files)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Both vectors should be L2-normalized.
    Lower distance means better match.
    """
    return 1.0 - float(np.dot(a, b))


def load_known_embeddings(app, images_dir: str, strip_trailing_numbers: bool) -> Dict[str, List[np.ndarray]]:
    images_root = Path(images_dir)

    if not images_root.exists():
        raise RuntimeError(f"Images folder does not exist: {images_root.resolve()}")

    image_files = find_image_files(images_root)

    if not image_files:
        raise RuntimeError(f"No image files found in: {images_root.resolve()}")

    known: Dict[str, List[np.ndarray]] = {}

    print(f"Loading known faces from: {images_root.resolve()}")
    print(f"Found {len(image_files)} image files.\n")

    for image_path in image_files:
        person_name = clean_person_name_from_filename(
            image_path,
            strip_trailing_numbers=strip_trailing_numbers,
        )

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] Could not read image: {image_path.name}")
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"[SKIP] No face detected in: {image_path.name}")
            continue

        if len(faces) > 1:
            print(f"[WARN] Multiple faces in {image_path.name}; using largest face.")

        largest_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        embedding = l2_normalize(largest_face.embedding.astype(np.float32))
        known.setdefault(person_name, []).append(embedding)

        print(f"[OK] {image_path.name} -> {person_name}")

    if not known:
        raise RuntimeError("No valid face embeddings were loaded.")

    print("\nLoaded people:")
    for name, embeddings in known.items():
        print(f"  {name}: {len(embeddings)} embedding(s)")

    print()
    return known


def recognize_face(
    live_embedding: np.ndarray,
    known_embeddings: Dict[str, List[np.ndarray]],
    threshold: float,
) -> Tuple[str, float]:
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
    label = f"{name} {distance:.3f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.putText(
        frame,
        label,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        color,
        2,
    )


def open_camera(camera_index: int, width: int, height: int):
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

    app = FaceAnalysis(
        name=args.model_pack,
        providers=["CPUExecutionProvider"],
    )

    app.prepare(
        ctx_id=-1,
        det_size=(args.det_size, args.det_size),
    )

    known_embeddings = load_known_embeddings(
        app,
        args.images,
        strip_trailing_numbers=not args.no_strip_trailing_numbers,
    )

    cap = open_camera(args.camera, args.camera_width, args.camera_height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try --camera 0 or --camera 1.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera resolution: {actual_w}x{actual_h}")
    print("Press ESC to exit.\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        faces = app.get(frame)

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

        cv2.imshow("InsightFace ArcFace - Your Images Folder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
