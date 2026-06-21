import csv
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
from pathlib import Path


def configure_tcl_tk_paths_for_windows() -> None:
    """Help Tkinter find Tcl/Tk on Windows Python installs.

    Some Windows installations, especially moved/copied Python folders or certain
    Python 3.13 installs, fail with:
        TclError: Can't find a usable init.tcl

    The Tcl/Tk files are normally still present under:
        <PythonRoot>\tcl\tcl8.6
        <PythonRoot>\tcl\tk8.6

    Tkinter reads TCL_LIBRARY and TK_LIBRARY before opening the first window, so
    we set them early if they are not already configured.
    """
    if os.name != "nt":
        return

    roots = []
    for raw in (
        Path(sys.executable).resolve().parent,
        Path(getattr(sys, "base_prefix", "") or sys.prefix),
        Path(sys.prefix),
    ):
        try:
            if raw and raw.exists():
                roots.append(raw)
        except Exception:
            pass

    # A venv executable lives under .venv\Scripts, while Tcl/Tk belongs to the
    # base Python installation. Still check parents because some portable installs
    # keep Tcl/Tk beside the venv or project.
    expanded_roots = []
    for root in roots:
        expanded_roots.extend([root, *list(root.parents)[:3]])

    seen = set()
    unique_roots = []
    for root in expanded_roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)

    tcl_candidates = []
    tk_candidates = []
    for root in unique_roots:
        tcl_candidates.extend([
            root / "tcl" / "tcl8.6",
            root / "tcl" / "tcl8.7",
            root / "lib" / "tcl8.6",
            root / "Lib" / "tcl8.6",
            root / "library",
        ])
        tk_candidates.extend([
            root / "tcl" / "tk8.6",
            root / "tcl" / "tk8.7",
            root / "lib" / "tk8.6",
            root / "Lib" / "tk8.6",
        ])

    if not os.environ.get("TCL_LIBRARY"):
        for candidate in tcl_candidates:
            if (candidate / "init.tcl").exists():
                os.environ["TCL_LIBRARY"] = str(candidate)
                break

    if not os.environ.get("TK_LIBRARY"):
        for candidate in tk_candidates:
            if (candidate / "tk.tcl").exists():
                os.environ["TK_LIBRARY"] = str(candidate)
                break


configure_tcl_tk_paths_for_windows()

# Self-bootstrap runtime packages and NVIDIA CUDA DLL search paths before
# importing cv2/Pillow/ONNX Runtime.  This makes `py desktop_station_app.py`
# repair the local Python environment automatically.
try:
    from faceattendance_runtime_bootstrap import ensure_faceattendance_runtime
    ensure_faceattendance_runtime(include_gui=True, prefer_gpu=True, verbose=True)
except Exception as _bootstrap_exc:
    print(f"[BOOTSTRAP] Runtime bootstrap warning: {_bootstrap_exc}")

import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except Exception as exc:  # pragma: no cover - displayed to the user in the GUI
    Image = None
    ImageTk = None
    PIL_IMPORT_ERROR = exc
else:
    PIL_IMPORT_ERROR = None

import moodle_yunet_sface_station as station

APP_DIR = Path(__file__).resolve().parent
UNKNOWN_DIR = APP_DIR / "unknown_review"
REVIEWED_EMBEDDINGS_DIR = APP_DIR / "reviewed_embeddings"
REPORTS_DIR = APP_DIR / "reports"


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_slug(text: str, fallback: str = "person") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip()).strip("_")
    return slug or fallback


def normalized_descriptor(values) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size != b.size:
        return -1.0
    return float(np.dot(a, b))


def crop_face_with_padding(frame: np.ndarray, box: Tuple[int, int, int, int], padding_ratio: float = 0.45) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = int(max(bw, bh) * padding_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2].copy()


class UnknownRegistry:
    """Local review database for unknown faces.

    Each unknown track is clustered by SFace descriptor similarity. Every third
    detection of the same cluster saves a face crop for later review.
    """

    def __init__(
        self,
        base_dir: Path = UNKNOWN_DIR,
        capture_every: int = 10,
        match_threshold: float = 0.50,
        capture_min_interval_seconds: float = 2.5,
        save_min_interval_seconds: float = 3.0,
    ):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.db_path = self.base_dir / "unknown_registry.json"
        self.capture_every = max(1, int(capture_every))
        self.match_threshold = float(match_threshold)
        self.capture_min_interval_seconds = float(capture_min_interval_seconds)
        self.save_min_interval_seconds = max(0.25, float(save_min_interval_seconds))
        self._last_save_at = 0.0
        self._dirty_updates = 0
        self.lock = threading.RLock()
        self.tracks: Dict[str, dict] = {}
        self.next_number = 1
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self) -> None:
        with self.lock:
            if not self.db_path.exists():
                return
            try:
                data = json.loads(self.db_path.read_text(encoding="utf-8"))
            except Exception:
                return
            self.tracks = {str(t["id"]): t for t in data.get("tracks", []) if isinstance(t, dict) and t.get("id")}
            self.next_number = int(data.get("next_number", 1))
            if self.tracks:
                max_seen = 0
                for tid in self.tracks:
                    m = re.search(r"(\d+)$", tid)
                    if m:
                        max_seen = max(max_seen, int(m.group(1)))
                self.next_number = max(self.next_number, max_seen + 1)

    def save(self, force: bool = True) -> None:
        with self.lock:
            now_ts = time.time()
            if not force:
                self._dirty_updates += 1
                if self._dirty_updates < 20 and (now_ts - self._last_save_at) < self.save_min_interval_seconds:
                    return
            data = {
                "version": 1,
                "updated_at": now_iso(),
                "capture_every": self.capture_every,
                "match_threshold": self.match_threshold,
                "next_number": self.next_number,
                "tracks": list(self.tracks.values()),
            }
            tmp_path = self.db_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.replace(self.db_path)
            self._last_save_at = now_ts
            self._dirty_updates = 0

    def _new_track(self, descriptor: np.ndarray) -> dict:
        track_id = f"UNK-{self.next_number:04d}"
        self.next_number += 1
        track = {
            "id": track_id,
            "created_at": now_iso(),
            "last_seen": now_iso(),
            "detections": 0,
            "captures": [],
            "assigned_label": "",
            "avg_descriptor": descriptor.astype(float).tolist(),
            "descriptors": [],
            "last_capture_time": 0.0,
        }
        self.tracks[track_id] = track
        return track

    def _find_track(self, descriptor: np.ndarray) -> Tuple[Optional[dict], float]:
        best_track = None
        best_score = -1.0
        for track in self.tracks.values():
            if track.get("assigned_label"):
                continue
            avg = normalized_descriptor(track.get("avg_descriptor"))
            score = cosine(descriptor, avg)
            if score > best_score:
                best_track = track
                best_score = score
        if best_track is not None and best_score >= self.match_threshold:
            return best_track, best_score
        return None, best_score

    def update_from_candidate(self, candidate: dict, frame: np.ndarray) -> Optional[dict]:
        descriptor = normalized_descriptor(candidate.get("descriptor"))
        if descriptor is None:
            return None

        with self.lock:
            track, match_score = self._find_track(descriptor)
            created_new_track = track is None
            if track is None:
                track = self._new_track(descriptor)

            track["detections"] = int(track.get("detections", 0)) + 1
            track["last_seen"] = now_iso()
            track["last_box"] = [int(v) for v in candidate.get("box", [0, 0, 0, 0])]
            track["last_similarity_to_track"] = round(float(match_score), 4) if match_score >= 0 else None

            descriptors = track.setdefault("descriptors", [])
            descriptors.append(descriptor.astype(float).tolist())
            if len(descriptors) > 20:
                del descriptors[:-20]

            old_avg = normalized_descriptor(track.get("avg_descriptor"))
            if old_avg is None:
                new_avg = descriptor
            else:
                # Low-pass update keeps the identity stable but lets it improve as more detections arrive.
                new_avg = normalized_descriptor((0.85 * old_avg) + (0.15 * descriptor))
                if new_avg is None:
                    new_avg = descriptor
            track["avg_descriptor"] = new_avg.astype(float).tolist()

            detection_no = int(track["detections"])
            elapsed = time.time() - float(track.get("last_capture_time", 0.0))
            should_capture = detection_no % self.capture_every == 0 and elapsed >= self.capture_min_interval_seconds
            if should_capture:
                crop = crop_face_with_padding(frame, candidate.get("box", [0, 0, 0, 0]))
                if crop.size > 0:
                    filename = f"{track['id']}_det{detection_no}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    image_path = self.images_dir / filename
                    cv2.imwrite(str(image_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    track.setdefault("captures", []).append({
                        "path": str(image_path.relative_to(self.base_dir)),
                        "created_at": now_iso(),
                        "detection_no": detection_no,
                    })
                    track["last_capture_time"] = time.time()

            self.save(force=created_new_track or should_capture)
            return dict(track)

    def assign_label(self, track_id: str, label: str) -> List[np.ndarray]:
        label = str(label or "").strip()
        if not label:
            raise ValueError("Label cannot be empty.")

        with self.lock:
            track = self.tracks.get(str(track_id))
            if not track:
                raise ValueError(f"Unknown track not found: {track_id}")
            track["assigned_label"] = label
            track["assigned_at"] = now_iso()

            descriptors = []
            for raw in track.get("descriptors", []):
                desc = normalized_descriptor(raw)
                if desc is not None:
                    descriptors.append(desc)
            avg = normalized_descriptor(track.get("avg_descriptor"))
            if avg is not None:
                descriptors.insert(0, avg)
            if not descriptors:
                raise ValueError("This unknown track has no usable descriptors.")

            REVIEWED_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "name": label,
                "model": {"family": "opencv", "detector": "yunet", "recognizer": "sface"},
                "created_at": now_iso(),
                "source": "desktop_unknown_review",
                "unknown_track_id": track_id,
                "embeddings": [d.astype(float).tolist() for d in descriptors[:10]],
            }
            out_path = REVIEWED_EMBEDDINGS_DIR / f"{safe_slug(label)}_from_{track_id}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            track["reviewed_embedding_file"] = str(out_path)
            self.save()
            return descriptors

    def delete_track(self, track_id: str, delete_images: bool = True) -> bool:
        with self.lock:
            track = self.tracks.pop(str(track_id), None)
            if not track:
                return False
            if delete_images:
                for capture in track.get("captures", []):
                    rel = capture.get("path")
                    if not rel:
                        continue
                    try:
                        image_path = self.base_dir / rel
                        if image_path.exists():
                            image_path.unlink()
                    except Exception:
                        pass
            self.save()
            return True

    def delete_unassigned_tracks(self, delete_images: bool = True) -> int:
        with self.lock:
            ids = [tid for tid, track in self.tracks.items() if not track.get("assigned_label")]
        deleted = 0
        for track_id in ids:
            if self.delete_track(track_id, delete_images=delete_images):
                deleted += 1
        return deleted


    def delete_tracks_without_captures(self, delete_images: bool = True, only_unassigned: bool = True) -> int:
        """Delete unknown review records that have no saved face pictures.

        By default it removes only unlabeled unknown tracks so a reviewed/assigned
        track is not accidentally lost just because it has no captured image yet.
        """
        with self.lock:
            ids = [
                tid
                for tid, track in self.tracks.items()
                if len(track.get("captures", []) or []) == 0
                and (not only_unassigned or not track.get("assigned_label"))
            ]
        deleted = 0
        for track_id in ids:
            if self.delete_track(track_id, delete_images=delete_images):
                deleted += 1
        return deleted

    def snapshot(self, sort_by: str = "last_seen") -> List[dict]:
        with self.lock:
            rows = [dict(t) for t in self.tracks.values()]
        if sort_by == "detections":
            rows.sort(key=lambda x: (int(x.get("detections", 0) or 0), x.get("last_seen", "")), reverse=True)
        elif sort_by == "captures":
            rows.sort(key=lambda x: (len(x.get("captures", [])), x.get("last_seen", "")), reverse=True)
        else:
            rows.sort(key=lambda x: x.get("last_seen", ""), reverse=True)
        return rows

    def latest_capture_path(self, track_id: str) -> Optional[Path]:
        with self.lock:
            track = self.tracks.get(str(track_id))
            if not track:
                return None
            captures = track.get("captures", [])
            if not captures:
                return None
            rel = captures[-1].get("path")
            if not rel:
                return None
            return self.base_dir / rel


def cv_dnn_backend_target(use_gpu: bool) -> Tuple[int, int, str]:
    """Return backend/target ids for OpenCV DNN-backed YuNet/SFace when available."""
    cpu_backend = getattr(cv2.dnn, "DNN_BACKEND_OPENCV", 3)
    cpu_target = getattr(cv2.dnn, "DNN_TARGET_CPU", 0)
    if not use_gpu:
        return cpu_backend, cpu_target, "CPU / OpenCV"

    cuda_backend = getattr(cv2.dnn, "DNN_BACKEND_CUDA", None)
    cuda_target = getattr(cv2.dnn, "DNN_TARGET_CUDA", None)
    if cuda_backend is None or cuda_target is None:
        return cpu_backend, cpu_target, "CPU fallback: OpenCV was built without DNN CUDA constants"

    cuda_devices = 0
    try:
        if hasattr(cv2, "cuda") and hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
            cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        cuda_devices = 0

    if cuda_devices <= 0:
        return cpu_backend, cpu_target, "CPU fallback: no CUDA-enabled OpenCV device detected"

    return int(cuda_backend), int(cuda_target), f"GPU / OpenCV CUDA ({cuda_devices} device(s))"


def create_detector_with_backend(use_gpu: bool):
    backend, target, label = cv_dnn_backend_target(use_gpu)
    try:
        detector = cv2.FaceDetectorYN_create(
            str(station.YUNET_MODEL),
            "",
            (320, 320),
            station.YUNET_SCORE_THRESHOLD,
            station.YUNET_NMS_THRESHOLD,
            station.YUNET_TOP_K,
            backend,
            target,
        )
        return detector, label
    except TypeError:
        detector = cv2.FaceDetectorYN_create(
            str(station.YUNET_MODEL),
            "",
            (320, 320),
            station.YUNET_SCORE_THRESHOLD,
            station.YUNET_NMS_THRESHOLD,
            station.YUNET_TOP_K,
        )
        fallback = "CPU fallback: this OpenCV build does not expose backend/target arguments for YuNet"
        return detector, fallback


def create_recognizer_with_backend(use_gpu: bool):
    if use_gpu and hasattr(station, "create_onnxruntime_sface_recognizer"):
        try:
            recognizer = station.create_onnxruntime_sface_recognizer(prefer_cuda=True)
            return recognizer, getattr(recognizer, "backend_label", "GPU / ONNX Runtime CUDA")
        except Exception as exc:
            print(f"[DNN] SFace ONNX Runtime unavailable: {exc}")

    backend, target, label = cv_dnn_backend_target(use_gpu)
    try:
        recognizer = cv2.FaceRecognizerSF_create(str(station.SFACE_MODEL), "", backend, target)
        return recognizer, label
    except TypeError:
        recognizer = cv2.FaceRecognizerSF_create(str(station.SFACE_MODEL), "")
        fallback = "CPU fallback: this OpenCV build does not expose backend/target arguments for SFace"
        return recognizer, fallback


@dataclass
class RuntimeOptions:
    source_kind: str
    source_value: str
    embeddings_source: str
    profile: str
    use_gpu: bool
    use_moodle: bool
    # 0 means: use the selected algorithm profile resolution.
    camera_width: int = 0
    camera_height: int = 0


class StationWorker(threading.Thread):
    def __init__(self, options: RuntimeOptions, frame_queue: queue.Queue, status_queue: queue.Queue, registry: UnknownRegistry):
        super().__init__(daemon=True)
        self.options = options
        self.frame_queue = frame_queue
        self.status_queue = status_queue
        self.registry = registry
        self.stop_event = threading.Event()
        self.controls_lock = threading.RLock()
        self.controls = {
            "zoom_enabled": False,
            "zoom_factor": 1.0,
            "zoom_center_x": 0.5,
            "zoom_center_y": 0.5,
            "draw_unknown": True,
            "grid_search": False,
            "hide_confirmed_known": True,
            "confirmed_similarity_threshold": station.CONFIRMED_KNOWN_SIMILARITY_THRESHOLD,
            "confirmed_stable_frames": station.CONFIRMED_KNOWN_STABLE_FRAMES,
            "hide_confirmed_unknown": station.HIDE_CONFIRMED_UNKNOWN_FACES,
            "confirmed_unknown_frames": station.CONFIRMED_UNKNOWN_STABLE_FRAMES,
            "unknown_track_match_threshold": station.UNKNOWN_TRACK_MATCH_THRESHOLD,
            "skip_resolved_recognition": station.SKIP_RESOLVED_FACE_RECOGNITION,
            "max_recognitions_per_frame": station.MAX_RECOGNITIONS_PER_FRAME,
            "known_stop_after_detections": station.KNOWN_STOP_AFTER_DETECTIONS,
        }
        self.known_features_lock = threading.RLock()
        self.known_features: Dict[str, List[np.ndarray]] = {}
        self.known_feature_index = station.build_known_feature_index(self.known_features)
        self.known_report: Dict[str, dict] = {}
        self.fps = 0.0
        self.cap = None

    def update_controls(self, **kwargs) -> None:
        with self.controls_lock:
            self.controls.update(kwargs)

    def register_known(self, label: str, descriptors: List[np.ndarray]) -> None:
        with self.known_features_lock:
            for descriptor in descriptors:
                desc = normalized_descriptor(descriptor)
                if desc is not None:
                    self.known_features.setdefault(label, []).append(desc.astype(np.float32))
            self.known_feature_index = station.build_known_feature_index(self.known_features)

    def request_stop(self) -> None:
        self.stop_event.set()

    def _status(self, text: str, **extra) -> None:
        payload = {"text": text, "time": now_iso()}
        payload.update(extra)
        try:
            self.status_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _put_frame(self, frame: np.ndarray, meta: dict) -> None:
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        try:
            self.frame_queue.put_nowait((frame, meta))
        except queue.Full:
            pass

    def _open_capture(self):
        source_kind = self.options.source_kind
        source_value = self.options.source_value.strip()

        if source_kind == "camera":
            try:
                index = int(source_value or "0")
            except ValueError:
                raise ValueError("Camera source must be a number, for example 0 or 1.")
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
            target_w = int(self.options.camera_width or station.CAMERA_WIDTH)
            target_h = int(self.options.camera_height or station.CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        # RTSP/HTTP stream or local video file.
        cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source_value)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _load_known_features(self, detector, recognizer) -> Dict[str, List[np.ndarray]]:
        if self.options.use_moodle:
            station.USE_MOODLE_API = True
            return station.load_known_features_from_moodle()

        station.USE_MOODLE_API = False
        station.EMBEDDINGS_SOURCE = self.options.embeddings_source or station.EMBEDDINGS_SOURCE
        return station.load_known_features_from_embeddings(station.EMBEDDINGS_SOURCE)

    def _update_station_controls(self) -> None:
        with self.controls_lock:
            controls = dict(self.controls)

        station.MANUAL_ZOOM_ENABLED = bool(controls.get("zoom_enabled"))
        station.MANUAL_ZOOM_FACTOR = float(controls.get("zoom_factor", 1.0))
        station.MANUAL_ZOOM_CENTER_X = float(controls.get("zoom_center_x", 0.5))
        station.MANUAL_ZOOM_CENTER_Y = float(controls.get("zoom_center_y", 0.5))
        station.DRAW_UNKNOWN_FACES = bool(controls.get("draw_unknown", True))
        station.ENABLE_PERIODIC_GRID_SEARCH = bool(controls.get("grid_search", False))
        station.HIDE_CONFIRMED_KNOWN_FACES = bool(controls.get("hide_confirmed_known", True))
        station.CONFIRMED_KNOWN_SIMILARITY_THRESHOLD = float(controls.get("confirmed_similarity_threshold", station.CONFIRMED_KNOWN_SIMILARITY_THRESHOLD))
        station.CONFIRMED_KNOWN_STABLE_FRAMES = int(float(controls.get("confirmed_stable_frames", station.CONFIRMED_KNOWN_STABLE_FRAMES)))
        station.HIDE_CONFIRMED_UNKNOWN_FACES = bool(controls.get("hide_confirmed_unknown", station.HIDE_CONFIRMED_UNKNOWN_FACES))
        station.CONFIRMED_UNKNOWN_STABLE_FRAMES = int(float(controls.get("confirmed_unknown_frames", station.CONFIRMED_UNKNOWN_STABLE_FRAMES)))
        station.UNKNOWN_TRACK_MATCH_THRESHOLD = float(controls.get("unknown_track_match_threshold", station.UNKNOWN_TRACK_MATCH_THRESHOLD))
        station.SKIP_RESOLVED_FACE_RECOGNITION = bool(controls.get("skip_resolved_recognition", station.SKIP_RESOLVED_FACE_RECOGNITION))
        station.MAX_RECOGNITIONS_PER_FRAME = max(0, int(float(controls.get("max_recognitions_per_frame", station.MAX_RECOGNITIONS_PER_FRAME))))
        station.KNOWN_STOP_AFTER_DETECTIONS = max(0, int(float(controls.get("known_stop_after_detections", station.KNOWN_STOP_AFTER_DETECTIONS))))

    def _draw_candidates(self, display_frame: np.ndarray, candidates: List[dict], face_state_by_name: dict, unknown_state_by_id: dict) -> None:
        h, w = display_frame.shape[:2]
        unknown_i = 0
        for candidate in candidates:
            if station.is_candidate_resolved_hidden(candidate, face_state_by_name, unknown_state_by_id):
                continue

            is_known = bool(candidate.get("is_known"))
            if not is_known and not station.DRAW_UNKNOWN_FACES:
                continue

            x1, y1, x2, y2 = [int(v) for v in candidate.get("box", [0, 0, 0, 0])]
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            color = (35, 190, 90) if is_known else (45, 70, 230)

            if is_known:
                raw_name = str(candidate.get("name", ""))
                label = station.short_display_name(station.display_name_for_person(raw_name))
                stable = face_state_by_name.get(raw_name, {}).get("stable_count", 0)
                label = f"{label} {min(stable, station.STABLE_FRAMES_REQUIRED)}/{station.STABLE_FRAMES_REQUIRED}"
            else:
                unknown_i += 1
                label = f"UNK{unknown_i}"
                if station.HIDE_CONFIRMED_UNKNOWN_FACES:
                    stable = int(candidate.get("unknown_stable_count", 0) or 0)
                    label += f" {min(stable, station.CONFIRMED_UNKNOWN_STABLE_FRAMES)}/{station.CONFIRMED_UNKNOWN_STABLE_FRAMES}"

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(display_frame, (x1, max(0, y1 - 28)), (min(w - 1, x1 + 12 * len(label) + 20), y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 8, max(18, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    def _ensure_known_report(self, name: str, candidate: Optional[dict] = None) -> dict:
        report = self.known_report.setdefault(name, {
            "name": station.display_name_for_person(name),
            "detections": 0,
            "first_seen": now_iso(),
            "last_seen": now_iso(),
            "best_similarity": -1.0,
            "attendance_marks": 0,
            "stopped": False,
            "stop_limit": station.KNOWN_STOP_AFTER_DETECTIONS,
        })
        report["stop_limit"] = int(station.KNOWN_STOP_AFTER_DETECTIONS or 0)
        if candidate is not None:
            try:
                report["best_similarity"] = max(float(report.get("best_similarity", -1.0)), float(candidate.get("similarity", -1.0)))
            except Exception:
                pass
        return report

    def _mark_known_report_stopped(self, name: str, candidate: Optional[dict] = None, update_last_seen: bool = True) -> dict:
        report = self._ensure_known_report(name, candidate)
        limit = int(station.KNOWN_STOP_AFTER_DETECTIONS or 0)
        if limit > 0:
            report["detections"] = min(int(report.get("detections", 0) or 0), limit)
            report["stopped"] = True
            report["stop_limit"] = limit
        if update_last_seen:
            report["last_seen"] = now_iso()
        return report

    def _known_report_has_reached_limit(self, name: str) -> bool:
        limit = int(station.KNOWN_STOP_AFTER_DETECTIONS or 0)
        if limit <= 0:
            return False
        report = self.known_report.get(name)
        return bool(report and int(report.get("detections", 0) or 0) >= limit)

    def _record_known_report(self, name: str, candidate: dict, attendance_marked: bool) -> None:
        report = self._ensure_known_report(name, candidate)
        limit = int(station.KNOWN_STOP_AFTER_DETECTIONS or 0)

        # This counter is the one shown in Reports > Known persons.  It is also
        # used as the hard stop counter, so never let it grow past the configured
        # limit.  Existing sessions that already went over the limit are clipped
        # back on the next UI refresh/known sighting.
        current = int(report.get("detections", 0) or 0)
        if limit > 0 and current >= limit:
            report["detections"] = limit
            report["stopped"] = True
        else:
            report["detections"] = current + 1
            if limit > 0 and int(report["detections"]) >= limit:
                report["detections"] = limit
                report["stopped"] = True

        report["stop_limit"] = limit
        report["last_seen"] = now_iso()
        if attendance_marked:
            report["attendance_marks"] += 1

    def _normalized_known_report_rows(self) -> List[dict]:
        limit = int(station.KNOWN_STOP_AFTER_DETECTIONS or 0)
        rows = []
        for row in self.known_report.values():
            row_copy = dict(row)
            if limit > 0 and int(row_copy.get("detections", 0) or 0) >= limit:
                row_copy["detections"] = limit
                row_copy["stopped"] = True
                row_copy["stop_limit"] = limit
            rows.append(row_copy)
        return rows

    def run(self) -> None:
        try:
            station.apply_algorithm_profile(self.options.profile or "fast_short", source="desktop")
            if int(self.options.camera_width or 0) > 0:
                station.CAMERA_WIDTH = int(self.options.camera_width)
            if int(self.options.camera_height or 0) > 0:
                station.CAMERA_HEIGHT = int(self.options.camera_height)
            station.ensure_model_file(station.YUNET_MODEL, station.YUNET_URL)
            station.ensure_model_file(station.SFACE_MODEL, station.SFACE_URL)
            station.check_opencv_api()

            detector, detector_backend = create_detector_with_backend(self.options.use_gpu)
            recognizer, recognizer_backend = create_recognizer_with_backend(self.options.use_gpu)
            backend_label = detector_backend if detector_backend == recognizer_backend else f"Detector: {detector_backend}; Recognizer: {recognizer_backend}"
            self._status("Loading known embeddings...", backend=backend_label)

            with self.known_features_lock:
                self.known_features = self._load_known_features(detector, recognizer)
                self.known_feature_index = station.build_known_feature_index(self.known_features)

            self.cap = self._open_capture()
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.options.source_value}")

            self._status("Station started", backend=backend_label, known_people=len(self.known_features))

            face_state_by_name = {}
            unknown_state_by_id = {"_next_number": 1}
            last_capture_time_by_name = {}
            saved_count_by_name = {}
            last_attendance_time_by_name = {}
            frame_index = 0
            last_candidates: List[dict] = []
            last_fps_time = time.time()
            frames_since_fps = 0
            last_report_snapshot_time = 0.0

            while not self.stop_event.is_set():
                loop_t0 = time.perf_counter()
                read_t0 = time.perf_counter()
                ok, frame = self.cap.read()
                read_ms = (time.perf_counter() - read_t0) * 1000.0
                if not ok or frame is None:
                    time.sleep(0.04)
                    continue

                self._update_station_controls()
                zoom_t0 = time.perf_counter()
                frame, _manual_zoom_crop = station.apply_manual_station_zoom(frame)
                zoom_ms = (time.perf_counter() - zoom_t0) * 1000.0

                now = time.time()
                frame_index += 1
                frames_since_fps += 1
                if now - last_fps_time >= 1.0:
                    self.fps = frames_since_fps / max(0.001, now - last_fps_time)
                    frames_since_fps = 0
                    last_fps_time = now

                with self.known_features_lock:
                    known_features_index = self.known_feature_index

                if self.options.use_moodle:
                    refreshed = station.refresh_moodle_state_if_needed(self.known_features)
                    if refreshed is not self.known_features:
                        with self.known_features_lock:
                            self.known_features = refreshed
                            self.known_feature_index = station.build_known_feature_index(self.known_features)
                            known_features_index = self.known_feature_index

                should_fast_search = frame_index % station.FAST_SEARCH_EVERY_N_FRAMES == 0
                should_full_grid_search = station.ENABLE_PERIODIC_GRID_SEARCH and frame_index % station.FULL_GRID_SEARCH_EVERY_N_FRAMES == 0
                candidates = last_candidates

                if should_fast_search or should_full_grid_search:
                    resolved_skip_targets = station.build_resolved_face_skip_targets(face_state_by_name, unknown_state_by_id)
                    candidates = station.detect_faces_with_search_zoom(
                        detector=detector,
                        recognizer=recognizer,
                        known_features=known_features_index,
                        frame=frame,
                        full_grid=should_full_grid_search,
                        last_box=None,
                        locked_name=None,
                        resolved_skip_targets=resolved_skip_targets,
                        frame_number=frame_index,
                    )
                    candidates = station.dedupe_candidates(candidates)
                    candidates = station.suppress_unknown_near_known(candidates)
                    last_candidates = candidates

                track_t0 = time.perf_counter()
                station.update_unknown_tracking(candidates, unknown_state_by_id, now)

                seen_known_names = set()
                unknown_registry_updates = 0
                unknown_registry_budget = max(0, int(getattr(station, "UNKNOWN_REGISTRY_MAX_UPDATES_PER_FRAME", 2)))
                for candidate in candidates:
                    if not candidate.get("is_known", False):
                        if (
                            unknown_registry_budget != 0
                            and unknown_registry_updates < unknown_registry_budget
                            and not station.is_candidate_confirmed_unknown_hidden(candidate, unknown_state_by_id)
                        ):
                            self.registry.update_from_candidate(candidate, frame)
                            unknown_registry_updates += 1
                        continue

                    name = str(candidate.get("name"))
                    if name in seen_known_names:
                        continue
                    seen_known_names.add(name)

                    state = face_state_by_name.setdefault(name, {
                        "stable_count": 0,
                        "last_seen": 0.0,
                        "last_box": None,
                        "last_similarity": -1.0,
                        "confirmed_hidden": False,
                        "confirmed_similarity": -1.0,
                        "confirmed_at": 0.0,
                        "recognition_count": 0,
                        "recognition_stopped": False,
                        "stopped_at": 0.0,
                    })
                    state["last_seen"] = now
                    state["last_box"] = candidate.get("box")
                    state["last_similarity"] = candidate.get("similarity", -1.0)

                    limit = int(station.KNOWN_STOP_AFTER_DETECTIONS or 0)
                    report_count = int(self.known_report.get(name, {}).get("detections", 0) or 0)
                    if limit > 0:
                        # Keep the internal state aligned with the exact counter visible in
                        # Reports > Known persons.  Without this, the UI counter could keep
                        # growing while the internal stop counter was a different value.
                        state["recognition_count"] = min(limit, max(int(state.get("recognition_count", 0) or 0), report_count))

                    if state.get("recognition_stopped", False) or self._known_report_has_reached_limit(name):
                        state["recognition_count"] = limit if limit > 0 else int(state.get("recognition_count", 0) or 0)
                        state["recognition_stopped"] = limit > 0
                        state["confirmed_hidden"] = True
                        self._mark_known_report_stopped(name, candidate)
                        continue

                    state["stable_count"] += 1
                    state["recognition_count"] = int(state.get("recognition_count", 0) or 0) + 1
                    station.mark_candidate_confirmed_if_ready(name, candidate, state, now)
                    station.mark_known_stopped_if_ready(name, candidate, state, now)

                    attendance_marked = False
                    if not state.get("recognition_stopped", False) and state["stable_count"] >= station.STABLE_FRAMES_REQUIRED and station.SAVE_PHOTO_WHEN_KNOWN_STABLE:
                        last_att = last_attendance_time_by_name.get(name, 0.0)
                        last_cap = last_capture_time_by_name.get(name, 0.0)
                        saved_count = saved_count_by_name.get(name, 0)
                        if (
                            now - last_att >= station.ATTENDANCE_COOLDOWN_SECONDS
                            and saved_count < station.MAX_PHOTOS_PER_PERSON
                            and now - last_cap >= station.PHOTO_COOLDOWN_SECONDS
                        ):
                            clean_zoomed = station.make_zoomed_proof_frame(frame, candidate["box"])
                            image_path = station.save_person_photo(station.display_name_for_person(name), clean_zoomed)
                            station.mark_attendance(name, image_path, candidate.get("similarity"))
                            last_capture_time_by_name[name] = now
                            last_attendance_time_by_name[name] = now
                            saved_count_by_name[name] = saved_count + 1
                            attendance_marked = True

                    self._record_known_report(name, candidate, attendance_marked)

                # Drop stale stability counters.
                for name in list(face_state_by_name.keys()):
                    if now - face_state_by_name[name]["last_seen"] > station.FACE_STATE_TIMEOUT_SECONDS:
                        del face_state_by_name[name]
                station.cleanup_unknown_tracking(unknown_state_by_id, now)
                track_ms = (time.perf_counter() - track_t0) * 1000.0

                should_output_frame = (frame_index % max(1, int(getattr(station, "DESKTOP_OUTPUT_EVERY_N_FRAMES", 1))) == 0)
                if not should_output_frame:
                    continue

                draw_t0 = time.perf_counter()
                display_frame = frame.copy()
                self._draw_candidates(display_frame, candidates, face_state_by_name, unknown_state_by_id)
                draw_ms = (time.perf_counter() - draw_t0) * 1000.0

                visible_candidates, hidden_confirmed_candidates = station.split_candidates_by_confirmation(candidates, face_state_by_name, unknown_state_by_id)
                known_count = sum(1 for c in visible_candidates if c.get("is_known", False))
                unknown_count = len(visible_candidates) - known_count
                hidden_confirmed_count = len(hidden_confirmed_candidates)
                status = (
                    f"FPS {self.fps:.1f} | faces {len(candidates)} | remaining {len(visible_candidates)}"
                    f" | known {known_count} | unknown {unknown_count}"
                )
                if hidden_confirmed_count:
                    status += f" | resolved hidden {hidden_confirmed_count}"
                status += f" | SFace {getattr(station, 'LAST_SFACE_CALLS', 0)}/frame"
                batches = int(getattr(station, 'LAST_SFACE_BATCHES', 0) or 0)
                if batches:
                    status += f"/{batches} batch"
                status += f" | {backend_label}"
                cv2.putText(display_frame, status, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                report_interval = float(getattr(station, "DESKTOP_REPORT_EVERY_SECONDS", 0.5) or 0.5)
                include_report = (now - last_report_snapshot_time) >= report_interval
                if include_report:
                    last_report_snapshot_time = now
                loop_ms = (time.perf_counter() - loop_t0) * 1000.0
                meta = {
                    "fps": self.fps,
                    "known_count": known_count,
                    "unknown_count": unknown_count,
                    "total_faces": len(candidates),
                    "remaining_faces": len(visible_candidates),
                    "hidden_confirmed_count": hidden_confirmed_count,
                    "backend": backend_label,
                    "sface_calls": int(getattr(station, "LAST_SFACE_CALLS", 0)),
                    "sface_batches": int(getattr(station, "LAST_SFACE_BATCHES", 0)),
                    "sface_backend": str(getattr(station, "LAST_SFACE_BACKEND", "")),
                    "detected_raw_faces": int(getattr(station, "LAST_DETECTED_FACES", 0)),
                    "detect_ms": float(getattr(station, "LAST_DETECT_MS", 0.0)),
                    "sface_ms": float(getattr(station, "LAST_SFACE_MS", 0.0)),
                    "sface_align_ms": float(getattr(station, "LAST_SFACE_ALIGN_MS", 0.0)),
                    "sface_infer_ms": float(getattr(station, "LAST_SFACE_INFER_MS", 0.0)),
                    "rec_total_ms": float(getattr(station, "LAST_RECOGNITION_TOTAL_MS", 0.0)),
                    "read_ms": read_ms,
                    "zoom_ms": zoom_ms,
                    "track_ms": track_ms,
                    "draw_ms": draw_ms,
                    "loop_ms": loop_ms,
                    "known_report": self._normalized_known_report_rows() if include_report else None,
                    "unknown_report": self.registry.snapshot() if include_report else None,
                }
                self._put_frame(display_frame, meta)

        except Exception as exc:
            self._status(f"ERROR: {exc}", error=traceback.format_exc())
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self._status("Station stopped")


class DesktopStationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FaceAttendance Desktop Station")
        self.geometry("1500x900")
        self.minsize(1180, 720)
        self.configure(bg="#101418")

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue = queue.Queue(maxsize=20)
        self.registry = UnknownRegistry()
        self.last_frame_ui_time = 0.0
        self.last_report_ui_time = 0.0
        self.latest_frame_meta: Optional[dict] = None
        self.unknown_sort_mode = "last_seen"
        self.unknown_sort_desc = True
        self.unknown_sort_field_var = tk.StringVar(value="Date")
        self.unknown_sort_direction_var = tk.StringVar(value="DESC")
        self.worker: Optional[StationWorker] = None
        self.last_photo = None
        self.preview_photo = None
        self.latest_meta = {}

        self._style()
        self._build_ui()
        self._poll_queues()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        if PIL_IMPORT_ERROR is not None:
            messagebox.showerror(
                "Missing Pillow",
                "The desktop app needs Pillow for video display. Install it with:\n\npy -m pip install pillow\n\n"
                f"Import error: {PIL_IMPORT_ERROR}",
            )

    def _style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#101418")
        style.configure("Card.TFrame", background="#161d24", relief="flat")
        style.configure("TLabel", background="#101418", foreground="#e9eef3", font=("Segoe UI", 10))
        style.configure("Muted.TLabel", foreground="#aab6c2", background="#101418")
        style.configure("Card.TLabel", background="#161d24", foreground="#e9eef3")
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 18), foreground="#ffffff", background="#101418")
        style.configure("Status.TLabel", foreground="#9be59b", background="#101418")
        style.configure("TButton", font=("Segoe UI Semibold", 10), padding=8)
        style.configure("Primary.TButton", font=("Segoe UI Semibold", 10), padding=8)
        style.configure("TCheckbutton", background="#161d24", foreground="#e9eef3", font=("Segoe UI", 10))
        style.configure("TNotebook", background="#101418", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI Semibold", 10))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10), background="#121820", fieldbackground="#121820", foreground="#e9eef3")
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="FaceAttendance Desktop Station", style="Title.TLabel").pack(side=tk.LEFT)

        # Keep the essential runtime controls outside the long settings panel.
        # They remain visible even when the right-side settings list overflows.
        actions = ttk.Frame(header)
        actions.pack(side=tk.RIGHT, padx=(12, 0))
        self.start_btn = ttk.Button(actions, text="Start", style="Primary.TButton", width=12, command=self.start_station)
        self.stop_btn = ttk.Button(actions, text="Stop", width=12, command=self.stop_station, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT, padx=(0, 12))

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        camera_tab = ttk.Frame(self.notebook)
        reports_tab = ttk.Frame(self.notebook)
        self.notebook.add(camera_tab, text="Camera Station")
        self.notebook.add(reports_tab, text="Reports & Unknown Review")

        self._build_camera_tab(camera_tab)
        self._build_reports_tab(reports_tab)


    def _create_scrollable_settings_panel(self, parent: ttk.Frame) -> ttk.Frame:
        """Create a right-side settings panel that can scroll vertically.

        The desktop UI can easily grow beyond the available height when crowd,
        zoom and unknown-review controls are all enabled.  A plain ttk.Frame gets
        clipped, which can hide critical controls on smaller displays.  The Start
        and Stop buttons are fixed in the top header; this panel is only for
        configuration.
        """
        outer = ttk.Frame(parent, style="Card.TFrame")
        outer.grid(row=0, column=1, sticky="ns")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(
            outer,
            width=380,
            bg="#161d24",
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")

        side = ttk.Frame(canvas, style="Card.TFrame", padding=14)
        window_id = canvas.create_window((0, 0), window=side, anchor="nw")

        def update_scroll_region(_event=None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_inner_width(event) -> None:
            canvas.itemconfigure(window_id, width=event.width)

        def on_mousewheel(event) -> None:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def bind_mousewheel(_event=None) -> None:
            canvas.bind_all("<MouseWheel>", on_mousewheel)

        def unbind_mousewheel(_event=None) -> None:
            canvas.unbind_all("<MouseWheel>")

        side.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_inner_width)
        outer.bind("<Enter>", bind_mousewheel)
        side.bind("<Enter>", bind_mousewheel)
        canvas.bind("<Leave>", unbind_mousewheel)
        outer.bind("<Leave>", unbind_mousewheel)

        return side

    def _build_camera_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=0)
        parent.rowconfigure(0, weight=1)

        video_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        video_card.rowconfigure(0, weight=1, minsize=520)
        video_card.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_card, text="Start the station to show the camera feed", anchor=tk.CENTER, style="Card.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.metrics_var = tk.StringVar(value="Faces: 0 | Known: 0 | Unknown: 0 | FPS: 0")
        ttk.Label(video_card, textvariable=self.metrics_var, style="Card.TLabel").grid(row=1, column=0, sticky="ew", pady=(10, 0))

        side = self._create_scrollable_settings_panel(parent)
        side.columnconfigure(1, weight=1)

        ttk.Label(side, text="Input source", style="Card.TLabel", font=("Segoe UI Semibold", 12)).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.source_kind = tk.StringVar(value="camera")
        ttk.Radiobutton(side, text="Camera index", variable=self.source_kind, value="camera").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(side, text="Network stream / video file", variable=self.source_kind, value="stream").grid(row=2, column=0, sticky="w", columnspan=2)

        ttk.Label(side, text="Source", style="Card.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))
        self.source_value = tk.StringVar(value="0")
        ttk.Entry(side, textvariable=self.source_value, width=36).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(side, text="Browse file", command=self.browse_source_file).grid(row=4, column=2, sticky="ew", padx=(6, 0), pady=(4, 0))

        ttk.Label(side, text="Embeddings source", style="Card.TLabel").grid(row=5, column=0, sticky="w", pady=(12, 0))
        self.embeddings_source = tk.StringVar(value=str(APP_DIR / "images"))
        ttk.Entry(side, textvariable=self.embeddings_source, width=36).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(side, text="Browse", command=self.browse_embeddings).grid(row=6, column=2, sticky="ew", padx=(6, 0), pady=(4, 0))

        ttk.Label(side, text="Profile", style="Card.TLabel").grid(row=7, column=0, sticky="w", pady=(12, 0))
        self.profile = tk.StringVar(value="walkthrough_realtime" if "walkthrough_realtime" in station.ALGORITHM_PROFILES else ("crowd_extreme" if "crowd_extreme" in station.ALGORITHM_PROFILES else ("crowd_turbo" if "crowd_turbo" in station.ALGORITHM_PROFILES else ("crowd_fast" if "crowd_fast" in station.ALGORITHM_PROFILES else "fast_short"))))
        profile_box = ttk.Combobox(side, textvariable=self.profile, state="readonly", values=list(station.ALGORITHM_PROFILES.keys()), width=34)
        profile_box.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        self.use_gpu = tk.BooleanVar(value=True)
        self.use_moodle = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Try GPU acceleration (ONNX Runtime CUDA for SFace)", variable=self.use_gpu).grid(row=9, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Checkbutton(side, text="Load roster/embeddings from Moodle API", variable=self.use_moodle).grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Separator(side).grid(row=11, column=0, columnspan=3, sticky="ew", pady=14)
        ttk.Label(side, text="Digital zoom", style="Card.TLabel", font=("Segoe UI Semibold", 12)).grid(row=12, column=0, columnspan=3, sticky="w")

        self.zoom_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Enable zoom before recognition", variable=self.zoom_enabled, command=self.push_controls).grid(row=13, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.zoom_factor = tk.DoubleVar(value=1.0)
        ttk.Label(side, text="Zoom factor", style="Card.TLabel").grid(row=14, column=0, sticky="w", pady=(10, 0))
        ttk.Scale(side, from_=1.0, to=8.0, variable=self.zoom_factor, command=lambda _v: self.push_controls()).grid(row=15, column=0, columnspan=3, sticky="ew")

        self.zoom_x = tk.DoubleVar(value=0.5)
        self.zoom_y = tk.DoubleVar(value=0.5)
        ttk.Label(side, text="Horizontal center", style="Card.TLabel").grid(row=16, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0.0, to=1.0, variable=self.zoom_x, command=lambda _v: self.push_controls()).grid(row=17, column=0, columnspan=3, sticky="ew")
        ttk.Label(side, text="Vertical center", style="Card.TLabel").grid(row=18, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0.0, to=1.0, variable=self.zoom_y, command=lambda _v: self.push_controls()).grid(row=19, column=0, columnspan=3, sticky="ew")

        pan = ttk.Frame(side, style="Card.TFrame")
        pan.grid(row=20, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        for c in range(3):
            pan.columnconfigure(c, weight=1)
        ttk.Button(pan, text="↑", command=lambda: self.pan_zoom(0, -0.06)).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(pan, text="←", command=lambda: self.pan_zoom(-0.06, 0)).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(pan, text="Reset", command=self.reset_zoom).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(pan, text="→", command=lambda: self.pan_zoom(0.06, 0)).grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(pan, text="↓", command=lambda: self.pan_zoom(0, 0.06)).grid(row=2, column=1, sticky="ew", padx=2, pady=2)

        self.draw_unknown = tk.BooleanVar(value=False)
        self.grid_search = tk.BooleanVar(value=False)
        self.hide_confirmed_known = tk.BooleanVar(value=True)
        self.confirmed_similarity = tk.DoubleVar(value=station.CONFIRMED_KNOWN_SIMILARITY_THRESHOLD)
        self.hide_confirmed_unknown = tk.BooleanVar(value=station.HIDE_CONFIRMED_UNKNOWN_FACES)
        self.confirmed_unknown_frames = tk.DoubleVar(value=station.CONFIRMED_UNKNOWN_STABLE_FRAMES)
        self.skip_resolved_recognition = tk.BooleanVar(value=station.SKIP_RESOLVED_FACE_RECOGNITION)
        self.max_recognitions_per_frame = tk.DoubleVar(value=24 if ("walkthrough_realtime" in station.ALGORITHM_PROFILES or "crowd_extreme" in station.ALGORITHM_PROFILES) else max(8, station.MAX_RECOGNITIONS_PER_FRAME))
        self.known_stop_after_detections = tk.DoubleVar(value=max(0, station.KNOWN_STOP_AFTER_DETECTIONS))
        ttk.Checkbutton(side, text="Draw unknown faces", variable=self.draw_unknown, command=self.push_controls).grid(row=21, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Checkbutton(side, text="Periodic grid search / slower high recall", variable=self.grid_search, command=self.push_controls).grid(row=22, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(side, text="Hide confirmed known faces", variable=self.hide_confirmed_known, command=self.push_controls).grid(row=23, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(side, text="Confirmed known similarity", style="Card.TLabel").grid(row=24, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0.36, to=0.80, variable=self.confirmed_similarity, command=lambda _v: self.push_controls()).grid(row=25, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(side, text="Stop repeated unknown faces", variable=self.hide_confirmed_unknown, command=self.push_controls).grid(row=26, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(side, text="Stop Unknown after detections", style="Card.TLabel").grid(row=27, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=3, to=30, variable=self.confirmed_unknown_frames, command=lambda _v: self.push_controls()).grid(row=28, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(side, text="Skip recognition for resolved faces", variable=self.skip_resolved_recognition, command=self.push_controls).grid(row=29, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(side, text="Max SFace recognitions/frame (0 = all)", style="Card.TLabel").grid(row=30, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0, to=40, variable=self.max_recognitions_per_frame, command=lambda _v: self.push_controls()).grid(row=31, column=0, columnspan=3, sticky="ew")
        ttk.Label(side, text="Stop known after detections", style="Card.TLabel").grid(row=32, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0, to=300, variable=self.known_stop_after_detections, command=lambda _v: self.push_controls()).grid(row=33, column=0, columnspan=3, sticky="ew")
        ttk.Label(side, text="0 = disabled. 150 is recommended for classroom tests.", style="Muted.TLabel").grid(row=34, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(
            side,
            text="Start/Stop controls are now fixed in the top bar.",
            style="Card.TLabel",
        ).grid(row=35, column=0, columnspan=3, sticky="w", pady=(18, 0))

    def _build_reports_tab(self, parent: ttk.Frame) -> None:
        """Build the reports page with Known on the left and Unknown review on the right."""
        parent.columnconfigure(0, weight=3, minsize=560)
        parent.columnconfigure(1, weight=2, minsize=430)
        parent.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(parent)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        top_bar.columnconfigure(0, weight=1)
        ttk.Label(top_bar, text="Reports & Unknown Review", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(top_bar, text="Refresh report", command=self.refresh_reports).grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Button(top_bar, text="Export CSV report", command=self.export_report_csv).grid(row=0, column=2, sticky="e", padx=(8, 0))

        known_section = ttk.Frame(parent)
        known_section.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        known_section.rowconfigure(1, weight=1)
        known_section.columnconfigure(0, weight=1)
        ttk.Label(known_section, text="Known persons", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))

        known_card = ttk.Frame(known_section, style="Card.TFrame", padding=10)
        known_card.grid(row=1, column=0, sticky="nsew")
        known_card.rowconfigure(0, weight=1)
        known_card.columnconfigure(0, weight=1)

        self.known_tree = ttk.Treeview(
            known_card,
            columns=("detections", "last_seen", "best", "marks", "status"),
            show="tree headings",
            height=14,
        )
        self.known_tree.heading("#0", text="Name")
        self.known_tree.heading("detections", text="Detections")
        self.known_tree.heading("last_seen", text="Last seen")
        self.known_tree.heading("best", text="Best sim")
        self.known_tree.heading("marks", text="Marks")
        self.known_tree.heading("status", text="Status")
        self.known_tree.column("#0", width=220, stretch=True)
        self.known_tree.column("detections", width=90, anchor=tk.CENTER, stretch=False)
        self.known_tree.column("last_seen", width=155, stretch=False)
        self.known_tree.column("best", width=85, anchor=tk.CENTER, stretch=False)
        self.known_tree.column("marks", width=70, anchor=tk.CENTER, stretch=False)
        self.known_tree.column("status", width=90, anchor=tk.CENTER, stretch=False)
        self.known_tree.grid(row=0, column=0, sticky="nsew")
        known_scroll = ttk.Scrollbar(known_card, orient=tk.VERTICAL, command=self.known_tree.yview)
        known_scroll.grid(row=0, column=1, sticky="ns")
        self.known_tree.configure(yscrollcommand=known_scroll.set)

        unknown_section = ttk.Frame(parent)
        unknown_section.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        unknown_section.rowconfigure(2, weight=1)
        unknown_section.columnconfigure(0, weight=1)
        ttk.Label(unknown_section, text="Unknown review", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))

        unknown_tools = ttk.Frame(unknown_section, style="Card.TFrame", padding=(10, 8))
        unknown_tools.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        unknown_tools.columnconfigure(1, weight=1)
        unknown_tools.columnconfigure(3, weight=1)

        ttk.Label(unknown_tools, text="Sort by", style="Card.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.unknown_sort_field_combo = ttk.Combobox(
            unknown_tools,
            textvariable=self.unknown_sort_field_var,
            state="readonly",
            values=("Date", "Detections"),
            width=14,
        )
        self.unknown_sort_field_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.unknown_sort_field_combo.bind("<<ComboboxSelected>>", self.on_unknown_sort_control_changed)

        ttk.Label(unknown_tools, text="Order", style="Card.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.unknown_sort_direction_combo = ttk.Combobox(
            unknown_tools,
            textvariable=self.unknown_sort_direction_var,
            state="readonly",
            values=("DESC", "ASC"),
            width=8,
        )
        self.unknown_sort_direction_combo.grid(row=0, column=3, sticky="ew")
        self.unknown_sort_direction_combo.bind("<<ComboboxSelected>>", self.on_unknown_sort_control_changed)

        ttk.Button(unknown_tools, text="Delete selected unknowns", command=self.delete_selected_unknown).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0), padx=(0, 4))
        ttk.Button(unknown_tools, text="Delete all unlabeled", command=self.delete_all_unlabeled_unknown).grid(row=1, column=2, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(unknown_tools, text="Delete records without pictures", command=self.delete_unknowns_without_pictures).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(7, 0))

        unknown_card = ttk.Frame(unknown_section, style="Card.TFrame", padding=10)
        unknown_card.grid(row=2, column=0, sticky="nsew")
        unknown_card.rowconfigure(0, weight=1)
        unknown_card.columnconfigure(0, weight=1)

        self.unknown_tree = ttk.Treeview(
            unknown_card,
            columns=("detections", "captures", "last_seen", "label"),
            show="tree headings",
            height=12,
            selectmode="extended",
        )
        self.unknown_tree.heading("#0", text="Track")
        self.unknown_tree.heading("detections", text="Detections", command=self.sort_unknown_by_detections)
        self.unknown_tree.heading("captures", text="Pictures")
        self.unknown_tree.heading("last_seen", text="Last seen", command=self.sort_unknown_by_date)
        self.unknown_tree.heading("label", text="Assigned label")
        self.unknown_tree.column("#0", width=90, stretch=False)
        self.unknown_tree.column("detections", width=80, anchor=tk.CENTER, stretch=False)
        self.unknown_tree.column("captures", width=70, anchor=tk.CENTER, stretch=False)
        self.unknown_tree.column("last_seen", width=145, stretch=False)
        self.unknown_tree.column("label", width=120, stretch=True)
        self.unknown_tree.grid(row=0, column=0, sticky="nsew")
        self.unknown_tree.bind("<<TreeviewSelect>>", self.on_unknown_selected)
        unknown_scroll = ttk.Scrollbar(unknown_card, orient=tk.VERTICAL, command=self.unknown_tree.yview)
        unknown_scroll.grid(row=0, column=1, sticky="ns")
        self.unknown_tree.configure(yscrollcommand=unknown_scroll.set)
        self._update_unknown_sort_headings()

        review_bar = ttk.Frame(unknown_card, style="Card.TFrame")
        review_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        review_bar.columnconfigure(1, weight=1)
        ttk.Label(review_bar, text="Assign selected unknown as", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.assign_label_var = tk.StringVar()
        ttk.Entry(review_bar, textvariable=self.assign_label_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(review_bar, text="Assign label", command=self.assign_unknown_label).grid(row=0, column=2, sticky="ew")

        preview_frame = ttk.Frame(unknown_card, style="Card.TFrame")
        preview_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.preview_label = ttk.Label(
            preview_frame,
            text="Select an unknown track to preview the last captured picture.",
            style="Card.TLabel",
            anchor=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def browse_source_file(self) -> None:
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")])
        if path:
            self.source_kind.set("stream")
            self.source_value.set(path)

    def browse_embeddings(self) -> None:
        path = filedialog.askdirectory(title="Select embeddings folder")
        if path:
            self.embeddings_source.set(path)

    def start_station(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        if PIL_IMPORT_ERROR is not None:
            messagebox.showerror("Missing Pillow", "Install Pillow first: py -m pip install pillow")
            return

        selected_profile = self.profile.get()
        # Apply the profile once in the UI thread before push_controls(), so
        # profile-specific stability values such as STABLE_FRAMES_REQUIRED=2 for
        # walk-through mode are not overwritten by stale defaults.
        try:
            station.apply_algorithm_profile(selected_profile, source="desktop-ui-controls")
        except Exception:
            pass

        options = RuntimeOptions(
            source_kind=self.source_kind.get(),
            source_value=self.source_value.get(),
            embeddings_source=self.embeddings_source.get(),
            profile=selected_profile,
            use_gpu=bool(self.use_gpu.get()),
            use_moodle=bool(self.use_moodle.get()),
        )
        self.worker = StationWorker(options, self.frame_queue, self.status_queue, self.registry)
        self.push_controls()
        self.worker.start()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.status_var.set("Starting...")

    def stop_station(self) -> None:
        if self.worker:
            self.worker.request_stop()
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)

    def push_controls(self) -> None:
        if self.worker:
            self.worker.update_controls(
                zoom_enabled=bool(self.zoom_enabled.get()),
                zoom_factor=float(self.zoom_factor.get()),
                zoom_center_x=float(self.zoom_x.get()),
                zoom_center_y=float(self.zoom_y.get()),
                draw_unknown=bool(self.draw_unknown.get()),
                grid_search=bool(self.grid_search.get()),
                hide_confirmed_known=bool(self.hide_confirmed_known.get()),
                confirmed_similarity_threshold=float(self.confirmed_similarity.get()),
                confirmed_stable_frames=station.STABLE_FRAMES_REQUIRED,
                hide_confirmed_unknown=bool(self.hide_confirmed_unknown.get()),
                confirmed_unknown_frames=max(1, int(round(float(self.confirmed_unknown_frames.get())))),
                skip_resolved_recognition=bool(self.skip_resolved_recognition.get()),
                max_recognitions_per_frame=max(0, int(round(float(self.max_recognitions_per_frame.get())))),
                known_stop_after_detections=max(0, int(round(float(self.known_stop_after_detections.get())))),
            )

    def pan_zoom(self, dx: float, dy: float) -> None:
        self.zoom_x.set(min(1.0, max(0.0, self.zoom_x.get() + dx)))
        self.zoom_y.set(min(1.0, max(0.0, self.zoom_y.get() + dy)))
        self.push_controls()

    def reset_zoom(self) -> None:
        self.zoom_factor.set(1.0)
        self.zoom_x.set(0.5)
        self.zoom_y.set(0.5)
        self.push_controls()

    def _poll_queues(self) -> None:
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.status_var.set(status.get("text", ""))
                if status.get("error"):
                    print(status["error"])
        except queue.Empty:
            pass

        try:
            frame, meta = self.frame_queue.get_nowait()
            self.latest_meta = meta
            self._show_frame(frame)
            self.metrics_var.set(
                f"Faces: {meta.get('total_faces', 0)} | Remaining: {meta.get('remaining_faces', meta.get('total_faces', 0))} | "
                f"Known: {meta.get('known_count', 0)} | Unknown: {meta.get('unknown_count', 0)} | "
                f"Hidden: {meta.get('hidden_confirmed_count', 0)} | "
                f"SFace: {meta.get('sface_calls', 0)}/{meta.get('sface_batches', 0)} | FPS: {meta.get('fps', 0.0):.1f} | "
                f"ms read {meta.get('read_ms', 0.0):.1f} det {meta.get('detect_ms', 0.0):.1f} "
                f"sface {meta.get('sface_ms', 0.0):.1f} align {meta.get('sface_align_ms', 0.0):.1f} "
                f"infer {meta.get('sface_infer_ms', 0.0):.1f} track {meta.get('track_ms', 0.0):.1f} "
                f"draw {meta.get('draw_ms', 0.0):.1f} loop {meta.get('loop_ms', 0.0):.1f}"
            )
            if meta.get("known_report") is not None or meta.get("unknown_report") is not None:
                self.refresh_reports(meta)
        except queue.Empty:
            pass

        if self.worker and not self.worker.is_alive():
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)

        self.after(33, self._poll_queues)

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        if Image is None or ImageTk is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        # Fill the available live-view area while preserving aspect ratio.
        # PIL thumbnail() only shrinks images; in the fast profiles the camera
        # frame can be 640x360, which made the live view look tiny inside a much
        # larger Tk label.  This resize intentionally allows upscaling for display
        # only; the recognition pipeline still uses the original frame.
        max_w = max(640, int(self.video_label.winfo_width() or 960))
        max_h = max(360, int(self.video_label.winfo_height() or 540))
        src_w, src_h = image.size
        if src_w > 0 and src_h > 0:
            scale = min(max_w / float(src_w), max_h / float(src_h))
            target = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
            resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)
            if target != image.size:
                image = image.resize(target, resample)

        self.last_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.last_photo, text="")

    def _sort_unknown_rows(self, rows: List[dict]) -> List[dict]:
        rows = list(rows or [])
        descending = bool(getattr(self, "unknown_sort_desc", True))
        mode = getattr(self, "unknown_sort_mode", "last_seen")
        if mode == "detections":
            rows.sort(
                key=lambda x: (
                    int(x.get("detections", 0) or 0),
                    str(x.get("last_seen", "") or ""),
                    str(x.get("id", "") or ""),
                ),
                reverse=descending,
            )
        else:
            rows.sort(
                key=lambda x: (
                    str(x.get("last_seen", "") or ""),
                    int(x.get("detections", 0) or 0),
                    str(x.get("id", "") or ""),
                ),
                reverse=descending,
            )
        return rows

    def _update_unknown_sort_headings(self) -> None:
        self._sync_unknown_sort_controls()
        if not hasattr(self, "unknown_tree"):
            return
        det_arrow = " ↓" if self.unknown_sort_mode == "detections" and self.unknown_sort_desc else " ↑" if self.unknown_sort_mode == "detections" else ""
        date_arrow = " ↓" if self.unknown_sort_mode == "last_seen" and self.unknown_sort_desc else " ↑" if self.unknown_sort_mode == "last_seen" else ""
        self.unknown_tree.heading("detections", text=f"Detections{det_arrow}", command=self.sort_unknown_by_detections)
        self.unknown_tree.heading("last_seen", text=f"Last seen{date_arrow}", command=self.sort_unknown_by_date)

    def refresh_reports(self, meta: Optional[dict] = None) -> None:
        meta = meta or self.latest_meta or {}
        known_rows = meta.get("known_report", [])
        unknown_rows = self.registry.snapshot(sort_by=self.unknown_sort_mode) if not meta.get("unknown_report") else list(meta.get("unknown_report", []))
        unknown_rows = self._sort_unknown_rows(unknown_rows)
        self._update_unknown_sort_headings()

        known_existing = set(self.known_tree.get_children())
        known_index = 0
        for row in known_rows:
            iid = safe_slug(row.get("name"), "known")
            values = (
                int(row.get("detections", 0)),
                row.get("last_seen", ""),
                f"{float(row.get('best_similarity', -1.0)):.3f}" if row.get("best_similarity", -1.0) is not None else "",
                int(row.get("attendance_marks", 0)),
                "Stopped" if row.get("stopped") else "Active",
            )
            if iid in known_existing:
                self.known_tree.item(iid, text=row.get("name", iid), values=values)
                self.known_tree.move(iid, "", known_index)
                known_existing.remove(iid)
            else:
                self.known_tree.insert("", known_index, iid=iid, text=row.get("name", iid), values=values)
            known_index += 1
        for iid in known_existing:
            self.known_tree.delete(iid)

        unknown_existing = set(self.unknown_tree.get_children())
        unknown_index = 0
        for row in unknown_rows:
            iid = str(row.get("id", ""))
            if not iid:
                continue
            values = (
                int(row.get("detections", 0)),
                len(row.get("captures", [])),
                row.get("last_seen", ""),
                row.get("assigned_label", ""),
            )
            if iid in unknown_existing:
                self.unknown_tree.item(iid, text=iid, values=values)
                self.unknown_tree.move(iid, "", unknown_index)
                unknown_existing.remove(iid)
            else:
                self.unknown_tree.insert("", unknown_index, iid=iid, text=iid, values=values)
            unknown_index += 1
        for iid in unknown_existing:
            self.unknown_tree.delete(iid)

    def _sync_unknown_sort_controls(self) -> None:
        if hasattr(self, "unknown_sort_field_var"):
            self.unknown_sort_field_var.set("Detections" if self.unknown_sort_mode == "detections" else "Date")
        if hasattr(self, "unknown_sort_direction_var"):
            self.unknown_sort_direction_var.set("DESC" if self.unknown_sort_desc else "ASC")

    def _sort_mode_from_controls(self) -> str:
        field = self.unknown_sort_field_var.get() if hasattr(self, "unknown_sort_field_var") else "Date"
        return "detections" if field == "Detections" else "last_seen"

    def _sort_desc_from_controls(self) -> bool:
        direction = self.unknown_sort_direction_var.get() if hasattr(self, "unknown_sort_direction_var") else "DESC"
        return str(direction).upper() == "DESC"

    def on_unknown_sort_control_changed(self, _event=None) -> None:
        self.set_unknown_sort(self._sort_mode_from_controls(), self._sort_desc_from_controls())

    def set_unknown_sort(self, mode: str, descending: bool) -> None:
        self.unknown_sort_mode = mode
        self.unknown_sort_desc = bool(descending)
        self._sync_unknown_sort_controls()
        self.refresh_reports()

    def sort_unknown_by_date_asc(self) -> None:
        self.set_unknown_sort("last_seen", False)

    def sort_unknown_by_date_desc(self) -> None:
        self.set_unknown_sort("last_seen", True)

    def sort_unknown_by_detections_asc(self) -> None:
        self.set_unknown_sort("detections", False)

    def sort_unknown_by_detections_desc(self) -> None:
        self.set_unknown_sort("detections", True)

    def sort_unknown_by_date(self) -> None:
        if self.unknown_sort_mode == "last_seen":
            self.unknown_sort_desc = not self.unknown_sort_desc
        else:
            self.unknown_sort_mode = "last_seen"
            self.unknown_sort_desc = True
        self.refresh_reports()

    def sort_unknown_by_detections(self) -> None:
        if self.unknown_sort_mode == "detections":
            self.unknown_sort_desc = not self.unknown_sort_desc
        else:
            self.unknown_sort_mode = "detections"
            self.unknown_sort_desc = True
        self.refresh_reports()

    def _force_unknown_report_refresh(self) -> None:
        if self.latest_meta is None:
            self.latest_meta = {}
        self.latest_meta["unknown_report"] = self._sort_unknown_rows(self.registry.snapshot(sort_by=self.unknown_sort_mode))
        self.refresh_reports()

    def delete_selected_unknown(self) -> None:
        selection = list(self.unknown_tree.selection())
        if not selection:
            messagebox.showwarning("No unknown selected", "Select one or more unknown tracks first.")
            return

        if len(selection) == 1:
            confirmation_text = f"Delete {selection[0]} and its saved pictures?"
        else:
            sample = ", ".join(selection[:6])
            if len(selection) > 6:
                sample += f", ... +{len(selection) - 6} more"
            confirmation_text = f"Delete {len(selection)} unknown tracks and their saved pictures?\n\n{sample}"

        if not messagebox.askyesno("Delete selected unknowns", confirmation_text):
            return

        deleted = 0
        missing = []
        for track_id in selection:
            if self.registry.delete_track(track_id, delete_images=True):
                deleted += 1
            else:
                missing.append(track_id)

        self.preview_label.configure(image="", text=f"Deleted {deleted} selected unknown track(s).")
        self._force_unknown_report_refresh()
        if missing:
            missing_text = ", ".join(missing[:8])
            if len(missing) > 8:
                missing_text += f", ... +{len(missing) - 8} more"
            messagebox.showwarning("Some unknowns were not found", f"Deleted {deleted}; not found: {missing_text}")
        else:
            messagebox.showinfo("Unknown cleanup", f"Deleted {deleted} selected unknown track(s).")

    def delete_all_unlabeled_unknown(self) -> None:
        if not messagebox.askyesno("Delete unlabeled unknowns", "Delete all unknown tracks that do not have an assigned label, including their saved pictures?"):
            return
        deleted = self.registry.delete_unassigned_tracks(delete_images=True)
        self.preview_label.configure(image="", text=f"Deleted {deleted} unlabeled unknown track(s).")
        self._force_unknown_report_refresh()
        messagebox.showinfo("Unknown cleanup", f"Deleted {deleted} unlabeled unknown track(s).")


    def delete_unknowns_without_pictures(self) -> None:
        if not messagebox.askyesno(
            "Delete unknowns without pictures",
            "Delete all unlabeled unknown records that have 0 saved pictures?\n\n"
            "This removes only review records without photos; labeled/assigned tracks are kept.",
        ):
            return
        deleted = self.registry.delete_tracks_without_captures(delete_images=True, only_unassigned=True)
        self.preview_label.configure(image="", text=f"Deleted {deleted} unknown record(s) without pictures.")
        self._force_unknown_report_refresh()
        messagebox.showinfo("Unknown cleanup", f"Deleted {deleted} unknown record(s) without pictures.")

    def on_unknown_selected(self, _event=None) -> None:
        selection = self.unknown_tree.selection()
        if not selection:
            return
        track_id = selection[0]
        path = self.registry.latest_capture_path(track_id)
        if not path or not path.exists() or Image is None or ImageTk is None:
            self.preview_label.configure(image="", text="No captured picture yet. It is saved every 3 detections of the same unknown track.")
            return
        image = Image.open(path)
        image.thumbnail((420, 240), Image.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def assign_unknown_label(self) -> None:
        selection = self.unknown_tree.selection()
        if not selection:
            messagebox.showwarning("No unknown selected", "Select an unknown track first.")
            return
        if len(selection) > 1:
            messagebox.showwarning("Multiple unknowns selected", "Assign label works with one unknown track at a time. Select only one track, or use Delete selected unknowns for multi-delete.")
            return
        label = self.assign_label_var.get().strip()
        if not label:
            messagebox.showwarning("Missing label", "Enter the person/student name to assign.")
            return
        track_id = selection[0]
        try:
            descriptors = self.registry.assign_label(track_id, label)
            if self.worker and self.worker.is_alive():
                self.worker.register_known(label, descriptors)
            self.refresh_reports()
            messagebox.showinfo("Label assigned", f"{track_id} was assigned to {label}. Future detections can now use this label in this running app.")
        except Exception as exc:
            messagebox.showerror("Could not assign label", str(exc))

    def export_report_csv(self) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = REPORTS_DIR / f"station_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        meta = self.latest_meta or {}
        known_rows = meta.get("known_report", [])
        unknown_rows = self.registry.snapshot(sort_by=self.unknown_sort_mode) if not meta.get("unknown_report") else list(meta.get("unknown_report", []))
        unknown_rows = self._sort_unknown_rows(unknown_rows)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["section", "id_or_name", "detections", "captures", "last_seen", "best_similarity", "attendance_marks", "assigned_label"])
            for row in known_rows:
                writer.writerow(["known", row.get("name", ""), row.get("detections", 0), "", row.get("last_seen", ""), row.get("best_similarity", ""), row.get("attendance_marks", 0), ""])
            for row in unknown_rows:
                writer.writerow(["unknown", row.get("id", ""), row.get("detections", 0), len(row.get("captures", [])), row.get("last_seen", ""), "", "", row.get("assigned_label", "")])
        messagebox.showinfo("Report exported", f"Report saved to:\n{out_path}")

    def on_close(self) -> None:
        self.stop_station()
        self.after(250, self.destroy)


if __name__ == "__main__":
    try:
        app = DesktopStationApp()
        app.mainloop()
    except tk.TclError as exc:
        message = str(exc)
        print("\nERROR: Tkinter could not start the desktop UI.")
        print(message)
        if "init.tcl" in message or "Tcl" in message:
            print("\nWindows fix:")
            print("1) Check that these files exist in your Python folder:")
            print(r"   C:\Users\<you>\AppData\Local\Programs\Python\Python313\tcl\tcl8.6\init.tcl")
            print(r"   C:\Users\<you>\AppData\Local\Programs\Python\Python313\tcl\tk8.6\tk.tcl")
            print("2) If they exist, set TCL_LIBRARY and TK_LIBRARY to those folders.")
            print("3) If they do not exist, rerun the Python installer, choose Modify, and enable Tcl/Tk and IDLE.")
            print("   Installing Python 3.12 with Tcl/Tk is also a reliable option for this app.")
        raise
