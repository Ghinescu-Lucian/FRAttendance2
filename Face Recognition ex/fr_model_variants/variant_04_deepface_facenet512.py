"""
Variant 04: DeepFace wrapper, default model Facenet512 + RetinaFace detector.

This variant is mostly for comparison, not for fastest real-time attendance.
DeepFace is convenient because you can switch model_name/detector_backend from
the command line, but it is usually slower than OpenCV SFace and InsightFace.

Install:
  py -m pip install deepface tensorflow opencv-python numpy

Run:
  py variant_04_deepface_facenet512.py --subjects-dir images

Try other models:
  py variant_04_deepface_facenet512.py --subjects-dir images --model-name ArcFace
  py variant_04_deepface_facenet512.py --subjects-dir images --model-name VGG-Face
  py variant_04_deepface_facenet512.py --subjects-dir images --detector-backend opencv
"""

from __future__ import annotations

import argparse
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


def import_deepface():
    try:
        from deepface import DeepFace
        return DeepFace
    except Exception as exc:
        raise RuntimeError(
            "Could not import deepface. Install it with:\n"
            "  py -m pip install deepface tensorflow opencv-python numpy"
        ) from exc


def choose_deepface_embedding(representations: list) -> np.ndarray | None:
    if not representations:
        return None

    # DeepFace.represent usually returns a list of dicts with an "embedding" key.
    item = representations[0]
    if isinstance(item, dict) and "embedding" in item:
        return l2_normalize(np.asarray(item["embedding"], dtype=np.float32))

    return None


def deepface_represent_image_path(
    DeepFace,
    image_path: Path,
    model_name: str,
    detector_backend: str,
) -> np.ndarray | None:
    reps = DeepFace.represent(
        img_path=str(image_path),
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=True,
        align=True,
    )
    return choose_deepface_embedding(reps)


def deepface_represent_face_crop(
    DeepFace,
    face_crop_rgb: np.ndarray,
    model_name: str,
) -> np.ndarray | None:
    reps = DeepFace.represent(
        img_path=face_crop_rgb,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    return choose_deepface_embedding(reps)


def normalize_face_crop(face_obj: dict) -> np.ndarray:
    face = np.asarray(face_obj["face"])
    if face.dtype != np.uint8:
        if face.max() <= 1.0:
            face = (face * 255.0).clip(0, 255).astype(np.uint8)
        else:
            face = face.clip(0, 255).astype(np.uint8)
    return face


def facial_area_to_box(area: dict) -> Tuple[int, int, int, int]:
    x = int(area.get("x", 0))
    y = int(area.get("y", 0))
    w = int(area.get("w", 0))
    h = int(area.get("h", 0))
    return x, y, x + w, y + h


def build_gallery(
    subjects_dir: Path,
    DeepFace,
    model_name: str,
    detector_backend: str,
) -> Dict[str, List[np.ndarray]]:
    if not subjects_dir.exists():
        raise RuntimeError(f"Subjects folder does not exist: {subjects_dir.resolve()}")

    image_files = find_image_files(subjects_dir)
    if not image_files:
        raise RuntimeError(f"No subject images found in: {subjects_dir.resolve()}")

    gallery: Dict[str, List[np.ndarray]] = {}

    print(f"Building DeepFace gallery from: {subjects_dir.resolve()}")
    print(f"Model: {model_name}, detector: {detector_backend}")
    print(f"Found {len(image_files)} image(s).\n")

    for image_path in image_files:
        person_name = clean_person_name_from_path(image_path, subjects_dir)

        try:
            embedding = deepface_represent_image_path(DeepFace, image_path, model_name, detector_backend)
        except Exception as exc:
            print(f"[SKIP] {image_path.name}: {exc}")
            continue

        if embedding is None:
            print(f"[SKIP] Could not create embedding: {image_path.name}")
            continue

        gallery.setdefault(person_name, []).append(embedding)
        print(f"[OK] {image_path.name} -> {person_name}")

    if not gallery:
        raise RuntimeError("No valid DeepFace embeddings were created.")

    return gallery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects-dir", default="images", help="Folder containing known subject images.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--model-name", default="Facenet512", help="Examples: Facenet512, ArcFace, VGG-Face.")
    parser.add_argument("--detector-backend", default="retinaface", help="Examples: retinaface, opencv, ssd, mtcnn, mediapipe.")
    parser.add_argument("--match-threshold", type=float, default=0.45, help="Cosine threshold. Tune per selected DeepFace model.")
    parser.add_argument("--process-every-n", type=int, default=5, help="DeepFace is slow; process only every Nth frame.")
    parser.add_argument("--window", default="Variant 04 - DeepFace")
    args = parser.parse_args()

    configure_opencv_threads()
    DeepFace = import_deepface()

    gallery = build_gallery(Path(args.subjects_dir), DeepFace, args.model_name, args.detector_backend)
    print_gallery(gallery)

    cap = open_camera(args.camera, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    fps = FpsCounter()
    frame_index = 0
    last_results = []

    print("Press ESC to exit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from camera.")
            break

        frame_index += 1

        if frame_index % args.process_every_n == 0:
            try:
                # DeepFace generally expects RGB images. OpenCV gives BGR.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted = DeepFace.extract_faces(
                    img_path=rgb,
                    detector_backend=args.detector_backend,
                    enforce_detection=False,
                    align=True,
                )
            except Exception as exc:
                print(f"[WARN] DeepFace frame error: {exc}")
                extracted = []

            results = []
            for face_obj in extracted:
                if not isinstance(face_obj, dict) or "face" not in face_obj:
                    continue

                area = face_obj.get("facial_area") or {}
                box = facial_area_to_box(area)

                face_crop_rgb = normalize_face_crop(face_obj)
                embedding = deepface_represent_face_crop(DeepFace, face_crop_rgb, args.model_name)

                if embedding is None:
                    continue

                name, score = best_cosine_match(embedding, gallery, args.match_threshold)
                results.append((box, name, score))

            last_results = results

        for box, name, score in last_results:
            draw_face(frame, box, name, score, score_prefix="cos=")

        current_fps = fps.update()
        cv2.putText(
            frame,
            f"DeepFace {args.model_name} | cached_faces={len(last_results)} | fps={current_fps:.1f} | ESC exit",
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
