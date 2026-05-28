"""
Variant 03: dlib / face_recognition library.

This is useful as a classic baseline. It is easy to understand, but usually
slower and weaker than InsightFace for far classroom faces.

Install on Windows can be annoying because dlib may require CMake/Visual Studio
Build Tools. If installation becomes painful, skip this variant and use
OpenCV SFace + InsightFace first.

Install:
  py -m pip install face_recognition opencv-python numpy

Run:
  py variant_03_face_recognition_dlib.py --subjects-dir images
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from fr_common import (
    best_l2_match,
    clean_person_name_from_path,
    configure_opencv_threads,
    draw_face,
    find_image_files,
    open_camera,
    print_gallery,
    print_header,
    FpsCounter,
)


def import_face_recognition():
    try:
        import face_recognition
        return face_recognition
    except Exception as exc:
        raise RuntimeError(
            "Could not import face_recognition. Install it with:\n"
            "  py -m pip install face_recognition opencv-python numpy\n"
            "On Windows, dlib compilation may require Visual Studio Build Tools."
        ) from exc



def import_dlib_or_none():
    try:
        import dlib
        return dlib
    except Exception:
        return None


def dlib_cuda_available() -> bool:
    dlib = import_dlib_or_none()
    if dlib is None:
        return False
    return bool(getattr(dlib, "DLIB_USE_CUDA", False))


def choose_model(device: str, requested_model: str) -> Tuple[str, str]:
    cuda_ok = dlib_cuda_available()

    if device == "cpu":
        return requested_model, "cpu"

    if device == "cuda":
        if cuda_ok:
            return "cnn", "cuda"
        print("[WARN] CUDA requested, but dlib was not compiled with CUDA. Using CPU.")
        return requested_model, "cpu"

    # auto
    if cuda_ok:
        return "cnn", "cuda"
    return requested_model, "cpu"


def build_gallery(subjects_dir: Path, face_recognition, model: str) -> Dict[str, List[np.ndarray]]:
    if not subjects_dir.exists():
        raise RuntimeError(f"Subjects folder does not exist: {subjects_dir.resolve()}")

    image_files = find_image_files(subjects_dir)
    if not image_files:
        raise RuntimeError(f"No subject images found in: {subjects_dir.resolve()}")

    gallery: Dict[str, List[np.ndarray]] = {}

    print(f"Building dlib/face_recognition gallery from: {subjects_dir.resolve()}")
    print(f"Found {len(image_files)} image(s).\n")

    for image_path in image_files:
        person_name = clean_person_name_from_path(image_path, subjects_dir)
        bgr = cv2.imread(str(image_path))

        if bgr is None:
            print(f"[SKIP] Could not read {image_path}")
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model=model)

        if not locations:
            print(f"[SKIP] No face found: {image_path.name}")
            continue

        # Use largest face in the subject image.
        def loc_area(loc: Tuple[int, int, int, int]) -> int:
            top, right, bottom, left = loc
            return max(1, right - left) * max(1, bottom - top)

        largest = max(locations, key=loc_area)
        encodings = face_recognition.face_encodings(rgb, known_face_locations=[largest])

        if not encodings:
            print(f"[SKIP] Could not encode face: {image_path.name}")
            continue

        gallery.setdefault(person_name, []).append(np.asarray(encodings[0], dtype=np.float32))
        print(f"[OK] {image_path.name} -> {person_name}")

    if not gallery:
        raise RuntimeError("No valid dlib embeddings were created.")

    return gallery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects-dir", default="images", help="Folder containing known subject images.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--model", default="hog", choices=["hog", "cnn"], help="hog=CPU faster, cnn=more accurate but much slower unless dlib CUDA is built.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="auto selects cnn+CUDA only if dlib was compiled with CUDA.")
    parser.add_argument("--gpu", action="store_true", help="Compatibility shortcut for --device cuda.")
    parser.add_argument("--match-threshold", type=float, default=0.55, help="Euclidean distance threshold. Lower=stricter. Common range: 0.50-0.60.")
    parser.add_argument("--resize", type=float, default=0.5, help="Scale camera frame before detection. 0.5 is faster.")
    parser.add_argument("--window", default="Variant 03 - dlib face_recognition")
    args = parser.parse_args()

    if args.gpu:
        args.device = "cuda"

    configure_opencv_threads()
    face_recognition = import_face_recognition()
    effective_model, effective_device = choose_model(args.device, args.model)

    dlib = import_dlib_or_none()
    print_header("Variant 03 - dlib / face_recognition")
    print(f"Requested device: {args.device}")
    print(f"Effective device: {effective_device}")
    print(f"dlib CUDA compiled: {getattr(dlib, 'DLIB_USE_CUDA', False) if dlib else 'unknown'}")
    print(f"Face detector model: {effective_model}")

    gallery = build_gallery(Path(args.subjects_dir), face_recognition, effective_model)
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

        if args.resize != 1.0:
            small = cv2.resize(frame, (0, 0), fx=args.resize, fy=args.resize)
        else:
            small = frame

        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small, model=effective_model)
        encodings = face_recognition.face_encodings(rgb_small, known_face_locations=locations)

        inv_scale = 1.0 / args.resize

        for loc, embedding in zip(locations, encodings):
            top, right, bottom, left = loc
            box = (
                int(left * inv_scale),
                int(top * inv_scale),
                int(right * inv_scale),
                int(bottom * inv_scale),
            )
            name, score = best_l2_match(np.asarray(embedding, dtype=np.float32), gallery, args.match_threshold)
            draw_face(frame, box, name, score, score_prefix="s=")

        current_fps = fps.update()
        cv2.putText(
            frame,
            f"dlib {effective_model} [{effective_device}] | faces={len(locations)} | fps={current_fps:.1f} | ESC exit",
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
