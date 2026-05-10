import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ============================================================
# FAST InsightFace / ArcFace test program for your images folder
# ============================================================
#
# Optimizations compared with the previous version:
# 1. Uses buffalo_s by default instead of buffalo_l.
# 2. Uses smaller detector size by default.
# 3. Uses lower camera resolution by default.
# 4. Runs recognition only every N frames.
# 5. Processes a resized frame, then maps boxes back to display frame.
# 6. By default recognizes only the largest face, which is usually enough
#    for an attendance entrance-camera prototype.
#
# Install:
#   py -m pip install insightface onnxruntime opencv-python numpy
#
# Run:
#   py test_insightface_arcface_fast.py
#
# If camera is 0:
#   py test_insightface_arcface_fast.py --camera 0
#
# Faster:
#   py test_insightface_arcface_fast.py --process-width 480 --det-size 320 --every-n-frames 3
#
# More accurate but slower:
#   py test_insightface_arcface_fast.py --model-pack buffalo_l --process-width 960 --det-size 640
# ============================================================


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNKNOWN_LABEL = "Unknown"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast InsightFace ArcFace recognition adapted for a flat images/ folder."
    )

    parser.add_argument("--images", default="images", help="Known face images folder. Default: images")
    parser.add_argument("--camera", type=int, default=1, help="Camera index. Default: 1")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width. Default: 640")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height. Default: 480")

    parser.add_argument(
        "--model-pack",
        default="buffalo_s",
        help="buffalo_s is faster; buffalo_l is more accurate but slower. Default: buffalo_s",
    )

    parser.add_argument(
        "--det-size",
        type=int,
        default=320,
        help="Detector input size. Lower is faster. Try 320 or 480. Default: 320",
    )

    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="Resize camera frame to this width before recognition. Lower is faster. Default: 640",
    )

    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=2,
        help="Run recognition every N frames. Higher is faster but less responsive. Default: 2",
    )

    parser.add_argument(
        "--max-faces",
        type=int,
        default=1,
        help="Maximum faces to recognize per processed frame. Default: 1",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.42,
        help="Cosine distance threshold. Lower = stricter. Default: 0.42",
    )

    parser.add_argument(
        "--no-strip-trailing-numbers",
        action="store_true",
        help="Do not group Person1.jpg and Person2.jpg as Person.",
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Show approximate FPS.",
    )

    return parser.parse_args()


def clean_person_name_from_filename(image_path: Path, strip_trailing_numbers: bool = True) -> str:
    name = image_path.stem.strip()

    if strip_trailing_numbers:
        name = re.sub(r"[\s_-]*\d+$", "", name).strip()

    return name


def find_image_files(images_dir: Path) -> List[Path]:
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
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


def resize_for_processing(frame, process_width: int):
    frame_h, frame_w = frame.shape[:2]

    if frame_w <= process_width:
        return frame, 1.0, 1.0

    scale = process_width / float(frame_w)
    new_w = process_width
    new_h = max(1, int(round(frame_h * scale)))

    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    scale_back_x = frame_w / float(new_w)
    scale_back_y = frame_h / float(new_h)

    return small, scale_back_x, scale_back_y


def draw_result(frame, result):
    x1, y1, x2, y2 = result["bbox"]
    name = result["name"]
    distance = result["distance"]

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


def recognize_frame(app, frame, known_embeddings, threshold, process_width, max_faces):
    small, sx, sy = resize_for_processing(frame, process_width)

    faces = app.get(small)

    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )

    if max_faces > 0:
        faces = faces[:max_faces]

    results = []

    for face in faces:
        name, distance = recognize_face(face.embedding, known_embeddings, threshold)

        x1, y1, x2, y2 = face.bbox.astype(int)

        # Map box from resized processing frame back to original frame.
        bbox = (
            int(x1 * sx),
            int(y1 * sy),
            int(x2 * sx),
            int(y2 * sy),
        )

        results.append({
            "bbox": bbox,
            "name": name,
            "distance": distance,
        })

    return results


def open_camera(camera_index: int, width: int, height: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Helps reduce lag on some cameras/backends.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def main():
    args = parse_args()

    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: insightface. Install with:\n"
            "py -m pip install insightface onnxruntime opencv-python numpy"
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
    print(f"Model pack: {args.model_pack}")
    print(f"det-size: {args.det_size}")
    print(f"process-width: {args.process_width}")
    print(f"every-n-frames: {args.every_n_frames}")
    print("Press ESC to exit.\n")

    frame_index = 0
    last_results = []

    fps_start = cv2.getTickCount()
    fps_frames = 0
    current_fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        frame_index += 1

        if frame_index % args.every_n_frames == 0:
            last_results = recognize_frame(
                app=app,
                frame=frame,
                known_embeddings=known_embeddings,
                threshold=args.threshold,
                process_width=args.process_width,
                max_faces=args.max_faces,
            )

        display_frame = frame.copy()

        for result in last_results:
            draw_result(display_frame, result)

        fps_frames += 1
        elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()

        if elapsed >= 1.0:
            current_fps = fps_frames / elapsed
            fps_start = cv2.getTickCount()
            fps_frames = 0

        status = (
            f"{args.model_pack} | det {args.det_size} | proc {args.process_width} | "
            f"skip {args.every_n_frames} | faces {len(last_results)}"
        )

        if args.show_fps:
            status += f" | FPS {current_fps:.1f}"

        cv2.putText(
            display_frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Fast InsightFace ArcFace", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
