#!/usr/bin/env python3
"""
Generate OpenCV SFace embeddings compatible with the Moodle/OpenCV station.

Supported sources:
  1) A normal image folder with one subfolder per identity:
       input_root/
         Person_A/*.jpg
         Person_B/*.png

  2) LFW downloaded automatically into one-subfolder-per-person layout:
       --download-lfw

Output format:
  One JSON file per identity:
    {
      "name": "Person_A",
      "model": {"family": "opencv", "detector": "yunet", "recognizer": "sface"},
      "embeddings": [[128 floats], ...]
    }

This format is accepted by your station loader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)

# Official LFW archive path historically used by scikit-learn / UMass.
# If this URL is unavailable in your environment, manually download LFW and use --input-folder.
LFW_TGZ_URL = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Stats:
    images_seen: int = 0
    detections_ok: int = 0
    detections_failed: int = 0
    files_written: int = 0
    embeddings_written: int = 0


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_. -]+", "_", name).strip().replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "UnknownPerson"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[OK] Already exists: {dest}")
        return
    print(f"[DOWNLOAD] {url}")
    print(f"           -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def ensure_models(models_dir: Path) -> Tuple[Path, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    yunet = models_dir / "face_detection_yunet_2023mar.onnx"
    sface = models_dir / "face_recognition_sface_2021dec.onnx"
    download_file(YUNET_URL, yunet)
    download_file(SFACE_URL, sface)
    return yunet, sface


def ensure_lfw(dataset_dir: Path) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    lfw_root = dataset_dir / "lfw"
    if lfw_root.exists() and any(lfw_root.iterdir()):
        print(f"[OK] LFW already extracted: {lfw_root}")
        return lfw_root

    tgz_path = dataset_dir / "lfw.tgz"
    download_file(LFW_TGZ_URL, tgz_path)

    print(f"[EXTRACT] {tgz_path}")
    with tarfile.open(tgz_path, "r:gz") as tar:
        def is_safe(member: tarfile.TarInfo) -> bool:
            target = (dataset_dir / member.name).resolve()
            return str(target).startswith(str(dataset_dir.resolve()))

        safe_members = [m for m in tar.getmembers() if is_safe(m)]
        tar.extractall(dataset_dir, members=safe_members)

    if not lfw_root.exists():
        raise RuntimeError(f"LFW extraction did not create expected folder: {lfw_root}")
    return lfw_root


def iter_identity_dirs(input_root: Path) -> List[Path]:
    dirs = [p for p in input_root.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda p: p.name.lower())


def iter_images(identity_dir: Path) -> List[Path]:
    return sorted(
        [p for p in identity_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )


def read_image_bgr(path: Path, max_side: int = 0) -> Optional[np.ndarray]:
    img = cv2.imread(str(path))
    if img is None:
        return None

    if max_side and max(img.shape[:2]) > max_side:
        h, w = img.shape[:2]
        scale = max_side / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img


def create_detector_and_recognizer(yunet_model: Path, sface_model: Path, score: float, nms: float, top_k: int):
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("Your OpenCV build does not expose cv2.FaceDetectorYN. Install/upgrade opencv-contrib-python.")
    if not hasattr(cv2, "FaceRecognizerSF"):
        raise RuntimeError("Your OpenCV build does not expose cv2.FaceRecognizerSF. Install/upgrade opencv-contrib-python.")

    detector = cv2.FaceDetectorYN.create(str(yunet_model), "", (320, 320), score, nms, top_k)
    recognizer = cv2.FaceRecognizerSF.create(str(sface_model), "")
    return detector, recognizer


def detect_faces(detector, img: np.ndarray) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    result = detector.detect(img)
    faces = result[1] if isinstance(result, tuple) else result
    if faces is None or len(faces) == 0:
        return None
    return faces


def choose_best_face(faces: np.ndarray) -> np.ndarray:
    # YuNet face row: [x, y, w, h, landmarks..., score]
    # Prefer high score and larger area. This avoids tiny background faces.
    areas = faces[:, 2] * faces[:, 3]
    scores = faces[:, 14] if faces.shape[1] > 14 else np.ones(len(faces), dtype=np.float32)
    combined = areas * np.maximum(scores, 0.01)
    return faces[int(np.argmax(combined))]


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


def extract_embedding(detector, recognizer, img: np.ndarray) -> Optional[List[float]]:
    faces = detect_faces(detector, img)
    if faces is None:
        return None
    face = choose_best_face(faces)
    aligned = recognizer.alignCrop(img, face)
    feature = recognizer.feature(aligned)
    feature = l2_normalize(feature)
    if feature.size != 128:
        raise RuntimeError(f"Unexpected SFace vector length: {feature.size}; expected 128")
    return feature.astype(float).tolist()


def write_identity_json(output_dir: Path, person_name: str, embeddings: List[List[float]], source_paths: List[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{safe_name(person_name)}_sface_embeddings.json"
    payload = {
        "version": 1,
        "name": person_name.replace("_", " "),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "family": "opencv",
            "detector": "yunet",
            "recognizer": "sface",
            "recognizerModel": "face_recognition_sface_2021dec.onnx",
            "descriptorLength": 128,
            "descriptorNormalized": True,
        },
        "source": "generated_sface_embedding_pool",
        "source_images": source_paths,
        "embeddings": embeddings,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return out_path


def generate_embeddings(
    input_root: Path,
    output_dir: Path,
    detector,
    recognizer,
    max_people: int,
    images_per_person: int,
    min_images_per_person: int,
    shuffle: bool,
    max_side: int,
    stats: Stats,
) -> None:
    identity_dirs = iter_identity_dirs(input_root)
    eligible = []
    for identity_dir in identity_dirs:
        imgs = iter_images(identity_dir)
        if len(imgs) >= min_images_per_person:
            eligible.append((identity_dir, imgs))

    if shuffle:
        random.shuffle(eligible)

    if max_people > 0:
        eligible = eligible[:max_people]

    print(f"[INFO] Input root: {input_root}")
    print(f"[INFO] Eligible identities: {len(eligible)}")
    print(f"[INFO] Output folder: {output_dir}")

    for idx, (identity_dir, images) in enumerate(eligible, 1):
        if shuffle:
            random.shuffle(images)
        selected = images[:images_per_person] if images_per_person > 0 else images

        embeddings: List[List[float]] = []
        source_paths: List[str] = []

        for img_path in selected:
            stats.images_seen += 1
            img = read_image_bgr(img_path, max_side=max_side)
            if img is None:
                print(f"[SKIP] Cannot read image: {img_path}")
                stats.detections_failed += 1
                continue
            try:
                emb = extract_embedding(detector, recognizer, img)
            except Exception as exc:
                print(f"[SKIP] {img_path}: {exc}")
                stats.detections_failed += 1
                continue
            if emb is None:
                stats.detections_failed += 1
                continue
            embeddings.append(emb)
            source_paths.append(str(img_path))
            stats.detections_ok += 1

        if embeddings:
            out = write_identity_json(output_dir, identity_dir.name, embeddings, source_paths)
            stats.files_written += 1
            stats.embeddings_written += len(embeddings)
            print(f"[{idx:04d}/{len(eligible):04d}] {identity_dir.name}: {len(embeddings)} embedding(s) -> {out.name}")
        else:
            print(f"[{idx:04d}/{len(eligible):04d}] {identity_dir.name}: no usable faces")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate OpenCV SFace embeddings for the Moodle/OpenCV station.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-folder", type=Path, help="Folder with one subfolder per person.")
    source.add_argument("--download-lfw", action="store_true", help="Download/extract LFW and use it as source.")

    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets"), help="Where to store downloaded datasets.")
    parser.add_argument("--output", type=Path, default=Path("embedding_pool_lfw"), help="Output embeddings folder.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Folder containing/downloading YuNet and SFace ONNX models.")
    parser.add_argument("--max-people", type=int, default=200, help="Maximum identities to export. Use 0 for all.")
    parser.add_argument("--images-per-person", type=int, default=5, help="Maximum source images/embeddings per identity. Use 0 for all.")
    parser.add_argument("--min-images-per-person", type=int, default=2, help="Ignore identities with fewer source images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle identities/images before selecting.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed when --shuffle is used.")
    parser.add_argument("--det-score", type=float, default=0.70, help="YuNet detection score threshold.")
    parser.add_argument("--nms", type=float, default=0.30, help="YuNet NMS threshold.")
    parser.add_argument("--top-k", type=int, default=5000, help="YuNet top_k.")
    parser.add_argument("--max-side", type=int, default=1200, help="Resize huge images before detection. 0 disables resizing.")

    args = parser.parse_args()
    random.seed(args.seed)

    input_root = ensure_lfw(args.dataset_dir) if args.download_lfw else args.input_folder
    if not input_root or not input_root.exists():
        raise RuntimeError(f"Input folder does not exist: {input_root}")

    yunet, sface = ensure_models(args.models_dir)
    detector, recognizer = create_detector_and_recognizer(yunet, sface, args.det_score, args.nms, args.top_k)

    stats = Stats()
    generate_embeddings(
        input_root=input_root,
        output_dir=args.output,
        detector=detector,
        recognizer=recognizer,
        max_people=args.max_people,
        images_per_person=args.images_per_person,
        min_images_per_person=args.min_images_per_person,
        shuffle=args.shuffle,
        max_side=args.max_side,
        stats=stats,
    )

    print("\n[DONE]")
    print(f"Images processed:       {stats.images_seen}")
    print(f"Detections successful:  {stats.detections_ok}")
    print(f"Detections failed:      {stats.detections_failed}")
    print(f"JSON files written:     {stats.files_written}")
    print(f"Embeddings written:     {stats.embeddings_written}")
    print(f"Use this folder in the station embeddings selector: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
