"""
Variant 02: InsightFace FaceAnalysis, usually buffalo_l/buffalo_s.

This is normally the strongest practical variant for your attendance scenario.
It uses ArcFace-style embeddings and can use GPU through onnxruntime-gpu.

CPU install:
  py -m pip install insightface onnxruntime opencv-python numpy

GPU install, after you already have a compatible NVIDIA/CUDA environment:
  py -m pip install insightface onnxruntime-gpu opencv-python numpy

Run:
  py variant_02_insightface_buffalo.py --subjects-dir images --model-name buffalo_l
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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


def import_insightface():
    try:
        from insightface.app import FaceAnalysis
        return FaceAnalysis
    except Exception as exc:
        raise RuntimeError(
            "Could not import insightface. Install it with:\n"
            "  py -m pip install insightface onnxruntime opencv-python numpy\n"
            "or for GPU:\n"
            "  py -m pip install insightface onnxruntime-gpu opencv-python numpy"
        ) from exc


def create_app(model_name: str, det_size: int, use_gpu: bool):
    FaceAnalysis = import_insightface()
    app = FaceAnalysis(name=model_name, providers=None)
    ctx_id = 0 if use_gpu else -1
    app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
    return app


def build_gallery(subjects_dir: Path, app) -> Dict[str, List[np.ndarray]]:
    if not subjects_dir.exists():
        raise RuntimeError(f"Subjects folder does not exist: {subjects_dir.resolve()}")

    image_files = find_image_files(subjects_dir)
    if not image_files:
        raise RuntimeError(f"No subject images found in: {subjects_dir.resolve()}")

    gallery: Dict[str, List[np.ndarray]] = {}

    print(f"Building InsightFace gallery from: {subjects_dir.resolve()}")
    print(f"Found {len(image_files)} image(s).\n")

    for image_path in image_files:
        person_name = clean_person_name_from_path(image_path, subjects_dir)
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] Could not read {image_path}")
            continue

        faces = app.get(image)
        if not faces:
            print(f"[SKIP] No face found: {image_path.name}")
            continue

        # Use largest face in the subject image.
        face = max(
            faces,
            key=lambda f: max(1.0, float(f.bbox[2] - f.bbox[0])) * max(1.0, float(f.bbox[3] - f.bbox[1])),
        )

        embedding = l2_normalize(face.embedding)
        gallery.setdefault(person_name, []).append(embedding)
        print(f"[OK] {image_path.name} -> {person_name}")

    if not gallery:
        raise RuntimeError("No valid InsightFace embeddings were created.")

    return gallery


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects-dir", default="images", help="Folder containing known subject images.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--model-name", default="buffalo_l", help="Examples: buffalo_l, buffalo_s.")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--gpu", action="store_true", help="Use ctx_id=0. Requires onnxruntime-gpu and compatible CUDA setup.")
    parser.add_argument("--match-threshold", type=float, default=0.35, help="Cosine similarity threshold. Tune on your classroom images.")
    parser.add_argument("--window", default="Variant 02 - InsightFace")
    args = parser.parse_args()

    configure_opencv_threads()
    app = create_app(args.model_name, args.det_size, args.gpu)

    gallery = build_gallery(Path(args.subjects_dir), app)
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

        faces = app.get(frame)

        for face in faces:
            embedding = l2_normalize(face.embedding)
            name, score = best_cosine_match(embedding, gallery, args.match_threshold)
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            draw_face(frame, (x1, y1, x2, y2), name, score, score_prefix="cos=")

        current_fps = fps.update()
        cv2.putText(
            frame,
            f"InsightFace {args.model_name} | faces={len(faces)} | fps={current_fps:.1f} | ESC exit",
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
