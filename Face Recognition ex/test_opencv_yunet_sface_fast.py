import argparse
import os
import re
import urllib.request
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UNKNOWN_LABEL = "Unknown"

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast OpenCV YuNet + SFace face recognition test."
    )
    parser.add_argument("--images", default="images", help="Known face images folder. Default: images")
    parser.add_argument("--camera", type=int, default=1, help="Camera index. Default: 1")
    parser.add_argument("--camera-width", type=int, default=640, help="Requested camera width. Default: 640")
    parser.add_argument("--camera-height", type=int, default=480, help="Requested camera height. Default: 480")
    parser.add_argument("--process-width", type=int, default=640, help="Resize frame to this width before recognition. Default: 640")
    parser.add_argument("--every-n-frames", type=int, default=1, help="Run recognition every N frames. Default: 1")
    parser.add_argument("--max-faces", type=int, default=1, help="Maximum faces to recognize. Default: 1")
    parser.add_argument("--score-threshold", type=float, default=0.75, help="YuNet face detector threshold. Default: 0.75")
    parser.add_argument("--similarity-threshold", type=float, default=0.38, help="SFace cosine similarity threshold. Try 0.35-0.45. Default: 0.38")
    parser.add_argument("--hold-known-frames", type=int, default=12, help="Keep last known label during blur. Default: 12")
    parser.add_argument("--history-size", type=int, default=6, help="Recent label history size. Default: 6")
    parser.add_argument("--min-votes", type=int, default=2, help="Minimum recent votes. Default: 2")
    parser.add_argument("--no-strip-trailing-numbers", action="store_true", help="Do not group Person1.jpg and Person2.jpg as Person.")
    parser.add_argument("--show-fps", action="store_true", help="Show approximate FPS.")
    return parser.parse_args()


def ensure_model_file(path: Path, url: str):
    MODELS_DIR.mkdir(exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading model: {path.name}")
    print(url)
    urllib.request.urlretrieve(url, str(path))
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Failed to download model: {path}")


def check_opencv_api():
    missing = []
    if not hasattr(cv2, "FaceDetectorYN_create"):
        missing.append("cv2.FaceDetectorYN_create")
    if not hasattr(cv2, "FaceRecognizerSF_create"):
        missing.append("cv2.FaceRecognizerSF_create")
    if missing:
        raise RuntimeError(
            "Your OpenCV build does not include the required face APIs:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\n\nInstall OpenCV contrib:\n"
            "  py -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python\n"
            "  py -m pip install opencv-contrib-python numpy\n"
        )


def clean_person_name_from_filename(image_path: Path, strip_trailing_numbers: bool = True) -> str:
    name = image_path.stem.strip()
    if strip_trailing_numbers:
        name = re.sub(r"[\s_-]*\d+$", "", name).strip()
    return name


def find_image_files(images_dir: Path) -> List[Path]:
    if not images_dir.exists():
        raise RuntimeError(f"Images folder does not exist: {images_dir.resolve()}")
    return sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def l2_normalize(feature: np.ndarray) -> np.ndarray:
    feature = feature.reshape(-1).astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature if norm == 0 else feature / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def largest_face(faces: np.ndarray):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def create_detector(input_size: Tuple[int, int], score_threshold: float):
    return cv2.FaceDetectorYN_create(
        str(YUNET_MODEL), "", input_size, score_threshold, 0.3, 5000
    )


def create_recognizer():
    return cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")


def detect_faces(detector, image: np.ndarray, score_threshold: float):
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    detector.setScoreThreshold(score_threshold)
    _, faces = detector.detect(image)
    if faces is None:
        return np.empty((0, 15), dtype=np.float32)
    return faces


def extract_feature(recognizer, image: np.ndarray, face: np.ndarray) -> np.ndarray:
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return l2_normalize(feature)


def load_known_features(detector, recognizer, images_dir: str, score_threshold: float, strip_trailing_numbers: bool) -> Dict[str, List[np.ndarray]]:
    images_root = Path(images_dir)
    image_files = find_image_files(images_root)
    if not image_files:
        raise RuntimeError(f"No image files found in: {images_root.resolve()}")

    known: Dict[str, List[np.ndarray]] = {}
    print(f"Loading known faces from: {images_root.resolve()}")
    print(f"Found {len(image_files)} image files.\n")

    for image_path in image_files:
        person_name = clean_person_name_from_filename(image_path, strip_trailing_numbers)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[SKIP] Could not read image: {image_path.name}")
            continue
        faces = detect_faces(detector, image, score_threshold)
        face = largest_face(faces)
        if face is None:
            print(f"[SKIP] No face detected in: {image_path.name}")
            continue
        feature = extract_feature(recognizer, image, face)
        known.setdefault(person_name, []).append(feature)
        print(f"[OK] {image_path.name} -> {person_name}")

    if not known:
        raise RuntimeError("No valid face features were loaded.")

    print("\nLoaded people:")
    for name, features in known.items():
        print(f"  {name}: {len(features)} feature(s)")
    print()
    return known


def recognize_feature(feature: np.ndarray, known_features: Dict[str, List[np.ndarray]], similarity_threshold: float) -> Tuple[str, float]:
    best_name = UNKNOWN_LABEL
    best_similarity = -1.0
    for person_name, features in known_features.items():
        for known_feature in features:
            sim = cosine_similarity(feature, known_feature)
            if sim > best_similarity:
                best_similarity = sim
                best_name = person_name
    if best_similarity < similarity_threshold:
        return UNKNOWN_LABEL, best_similarity
    return best_name, best_similarity


def resize_for_processing(frame: np.ndarray, process_width: int):
    frame_h, frame_w = frame.shape[:2]
    if frame_w <= process_width:
        return frame, 1.0, 1.0
    scale = process_width / float(frame_w)
    new_w = process_width
    new_h = max(1, int(round(frame_h * scale)))
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small, frame_w / float(new_w), frame_h / float(new_h)


def scale_face_box(face: np.ndarray, sx: float, sy: float):
    x, y, w, h = face[:4]
    return int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)


def recognize_frame(detector, recognizer, frame: np.ndarray, known_features: Dict[str, List[np.ndarray]], process_width: int, score_threshold: float, similarity_threshold: float, max_faces: int):
    small, sx, sy = resize_for_processing(frame, process_width)
    faces = detect_faces(detector, small, score_threshold)
    if len(faces) == 0:
        return []
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    if max_faces > 0:
        faces = faces[:max_faces]

    results = []
    for face in faces:
        feature = extract_feature(recognizer, small, face)
        name, similarity = recognize_feature(feature, known_features, similarity_threshold)
        results.append({
            "bbox": scale_face_box(face, sx, sy),
            "raw_name": name,
            "name": name,
            "similarity": similarity,
        })
    return results


class IdentityStabilizer:
    def __init__(self, history_size=6, min_votes=2, hold_known_frames=12):
        self.history = deque(maxlen=history_size)
        self.min_votes = min_votes
        self.hold_known_frames = hold_known_frames
        self.last_known_name = UNKNOWN_LABEL
        self.last_known_similarity = -1.0
        self.frames_since_known = 999

    def update(self, raw_name: str, similarity: float):
        if raw_name != UNKNOWN_LABEL:
            self.history.append(raw_name)
            self.last_known_name = raw_name
            self.last_known_similarity = similarity
            self.frames_since_known = 0
            return raw_name, similarity, "LIVE"

        self.frames_since_known += 1
        self.history.append(UNKNOWN_LABEL)

        counts = {}
        for label in self.history:
            if label != UNKNOWN_LABEL:
                counts[label] = counts.get(label, 0) + 1
        if counts:
            voted_name, votes = max(counts.items(), key=lambda item: item[1])
            if votes >= self.min_votes and self.frames_since_known <= self.hold_known_frames:
                return voted_name, self.last_known_similarity, "VOTE"
        if self.last_known_name != UNKNOWN_LABEL and self.frames_since_known <= self.hold_known_frames:
            return self.last_known_name, self.last_known_similarity, "HOLD"
        self.last_known_name = UNKNOWN_LABEL
        self.last_known_similarity = -1.0
        return UNKNOWN_LABEL, similarity, "UNKNOWN"


def draw_result(frame: np.ndarray, result: dict):
    x1, y1, x2, y2 = result["bbox"]
    name = result["name"]
    similarity = result["similarity"]
    mode = result.get("mode", "")
    color = (0, 255, 0) if name != UNKNOWN_LABEL else (0, 0, 255)
    label = f"{name} {similarity:.3f}"
    if mode in ("HOLD", "VOTE"):
        label += f" [{mode}]"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)


def open_camera(camera_index: int, width: int, height: int):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    args = parse_args()
    check_opencv_api()
    ensure_model_file(YUNET_MODEL, YUNET_URL)
    ensure_model_file(SFACE_MODEL, SFACE_URL)

    detector = create_detector((320, 320), args.score_threshold)
    recognizer = create_recognizer()

    known_features = load_known_features(
        detector=detector,
        recognizer=recognizer,
        images_dir=args.images,
        score_threshold=args.score_threshold,
        strip_trailing_numbers=not args.no_strip_trailing_numbers,
    )

    cap = open_camera(args.camera, args.camera_width, args.camera_height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}. Try --camera 0 or --camera 1.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print(f"process-width: {args.process_width}")
    print(f"every-n-frames: {args.every_n_frames}")
    print(f"score-threshold: {args.score_threshold}")
    print(f"similarity-threshold: {args.similarity_threshold}")
    print("Press ESC to exit.\n")

    stabilizer = IdentityStabilizer(args.history_size, args.min_votes, args.hold_known_frames)
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
            results = recognize_frame(
                detector, recognizer, frame, known_features,
                args.process_width, args.score_threshold,
                args.similarity_threshold, args.max_faces,
            )
            if results:
                raw_name = results[0]["raw_name"]
                raw_similarity = results[0]["similarity"]
                stable_name, stable_similarity, mode = stabilizer.update(raw_name, raw_similarity)
                results[0]["name"] = stable_name
                results[0]["similarity"] = stable_similarity
                results[0]["mode"] = mode
                last_results = results
            else:
                stable_name, stable_similarity, mode = stabilizer.update(UNKNOWN_LABEL, -1.0)
                if stable_name != UNKNOWN_LABEL and last_results:
                    last_results[0]["name"] = stable_name
                    last_results[0]["similarity"] = stable_similarity
                    last_results[0]["mode"] = mode
                else:
                    last_results = []

        display_frame = frame.copy()
        for result in last_results:
            draw_result(display_frame, result)

        fps_frames += 1
        elapsed = (cv2.getTickCount() - fps_start) / cv2.getTickFrequency()
        if elapsed >= 1.0:
            current_fps = fps_frames / elapsed
            fps_start = cv2.getTickCount()
            fps_frames = 0

        status = f"YuNet+SFace | sim {args.similarity_threshold:.2f} | proc {args.process_width} | skip {args.every_n_frames}"
        if args.show_fps:
            status += f" | FPS {current_fps:.1f}"
        cv2.putText(display_frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Fast OpenCV YuNet + SFace", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
