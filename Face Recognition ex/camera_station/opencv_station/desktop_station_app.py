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
        capture_every: int = 3,
        match_threshold: float = 0.50,
        capture_min_interval_seconds: float = 1.5,
    ):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "images"
        self.db_path = self.base_dir / "unknown_registry.json"
        self.capture_every = max(1, int(capture_every))
        self.match_threshold = float(match_threshold)
        self.capture_min_interval_seconds = float(capture_min_interval_seconds)
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

    def save(self) -> None:
        with self.lock:
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

            self.save()
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

    def snapshot(self) -> List[dict]:
        with self.lock:
            return [dict(t) for t in sorted(self.tracks.values(), key=lambda x: x.get("last_seen", ""), reverse=True)]

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
    camera_width: int = 1280
    camera_height: int = 720
    stream_max_width: int = 960


class ResilientNetworkCapture:
    """Latest-frame RTSP/HTTP/video-file reader with reconnect support.

    OpenCV's VideoCapture.read() can block for a long time on RTSP streams.
    This wrapper keeps the blocking read in a daemon thread. The recognition
    worker receives only the newest fresh frame, so old buffered frames are
    dropped instead of being processed seconds late.
    """

    def __init__(self, source: str, status_callback=None, stale_seconds: float = 2.5):
        self.source = str(source or "").strip()
        self.status_callback = status_callback
        self.stale_seconds = float(stale_seconds)
        self._stop_event = threading.Event()
        self._frame_lock = threading.RLock()
        self._last_frame = None
        self._last_frame_time = 0.0
        self._last_frame_id = 0
        self._last_consumed_id = 0
        self._cap = None
        self._last_status_time = 0.0
        self._open_attempts = 0
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _status(self, text: str, min_interval: float = 1.5) -> None:
        if self.status_callback is None:
            return
        now = time.time()
        if now - self._last_status_time < min_interval:
            return
        self._last_status_time = now
        try:
            self.status_callback(text)
        except Exception:
            pass

    def _configure_ffmpeg_options(self) -> None:
        # Advanced override for special cameras. If the user set the variable, do not replace it.
        if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
            return
        if self.source.lower().startswith("rtsp://"):
            # Lower latency and a finite timeout. TCP is usually more stable for Hikvision/HiLook LAN cameras.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000|max_delay;500000|fflags;nobuffer|flags;low_delay"
            )

    def _open_once(self):
        self._configure_ffmpeg_options()
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(self.source)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _sleep_interruptible(self, seconds: float) -> None:
        end = time.time() + max(0.0, seconds)
        while not self._stop_event.is_set() and time.time() < end:
            time.sleep(min(0.05, end - time.time()))

    def _reader_loop(self) -> None:
        reconnect_delay = 0.5
        while not self._stop_event.is_set():
            self._open_attempts += 1
            self._status("Connecting network stream...")
            cap = self._open_once()
            self._cap = cap

            if not cap.isOpened():
                self._status("Network stream could not be opened. Retrying...")
                try:
                    cap.release()
                except Exception:
                    pass
                self._sleep_interruptible(min(5.0, reconnect_delay))
                reconnect_delay = min(5.0, reconnect_delay * 1.5)
                continue

            self._status("Network stream connected")
            reconnect_delay = 0.5
            failures = 0

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 5:
                        self._status("Network stream lost. Reconnecting...")
                        break
                    self._sleep_interruptible(0.02)
                    continue

                failures = 0
                with self._frame_lock:
                    self._last_frame = frame
                    self._last_frame_time = time.time()
                    self._last_frame_id += 1

            try:
                cap.release()
            except Exception:
                pass
            self._cap = None
            self._sleep_interruptible(min(3.0, reconnect_delay))
            reconnect_delay = min(5.0, reconnect_delay * 1.3)

    def isOpened(self) -> bool:
        # The wrapper is usable while its reader thread is alive; the first frame may arrive a moment later.
        return self._thread.is_alive() and not self._stop_event.is_set()

    def read(self):
        with self._frame_lock:
            if self._last_frame is None:
                return False, None
            age = time.time() - self._last_frame_time
            if age > self.stale_seconds:
                self._status("Waiting for fresh network frames / reconnecting...")
                return False, None
            if self._last_frame_id == self._last_consumed_id:
                # Do not process the same RTSP frame repeatedly.
                return False, None
            self._last_consumed_id = self._last_frame_id
            return True, self._last_frame.copy()

    def release(self) -> None:
        self._stop_event.set()
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


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
        }
        self.controls_version = 0
        self.known_features_lock = threading.RLock()
        self.known_features: Dict[str, List[np.ndarray]] = {}
        self.known_report: Dict[str, dict] = {}
        self.fps = 0.0
        self.cap = None

    def update_controls(self, **kwargs) -> None:
        with self.controls_lock:
            changed = False
            for key, value in kwargs.items():
                if self.controls.get(key) != value:
                    changed = True
                self.controls[key] = value
            if changed:
                self.controls_version += 1

    def register_known(self, label: str, descriptors: List[np.ndarray]) -> None:
        with self.known_features_lock:
            for descriptor in descriptors:
                desc = normalized_descriptor(descriptor)
                if desc is not None:
                    self.known_features.setdefault(label, []).append(desc.astype(np.float32))

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
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.options.camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.options.camera_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap

        # RTSP/HTTP stream or local video file. Use a threaded latest-frame reader
        # so the main recognition loop never blocks inside VideoCapture.read().
        return ResilientNetworkCapture(source_value, status_callback=lambda text: self._status(text))

    def _load_known_features(self, detector, recognizer) -> Dict[str, List[np.ndarray]]:
        if self.options.use_moodle:
            station.USE_MOODLE_API = True
            return station.load_known_features_from_moodle()

        station.USE_MOODLE_API = False
        station.EMBEDDINGS_SOURCE = self.options.embeddings_source or station.EMBEDDINGS_SOURCE
        return station.load_known_features_from_embeddings(station.EMBEDDINGS_SOURCE)

    def _update_station_controls(self) -> Tuple[dict, int]:
        with self.controls_lock:
            controls = dict(self.controls)
            version = int(self.controls_version)

        zoom_factor = max(1.0, float(controls.get("zoom_factor", 1.0)))
        zoom_enabled = bool(controls.get("zoom_enabled")) and zoom_factor > 1.001

        # The desktop UI goes to 8x. Keep the station-side clamp in sync;
        # otherwise values above the old default max are silently clamped.
        station.MANUAL_ZOOM_MAX = max(float(getattr(station, "MANUAL_ZOOM_MAX", 5.0)), 8.0)
        station.MANUAL_ZOOM_ENABLED = zoom_enabled
        station.MANUAL_ZOOM_FACTOR = zoom_factor
        station.MANUAL_ZOOM_CENTER_X = float(controls.get("zoom_center_x", 0.5))
        station.MANUAL_ZOOM_CENTER_Y = float(controls.get("zoom_center_y", 0.5))
        station.DRAW_UNKNOWN_FACES = bool(controls.get("draw_unknown", True))
        station.ENABLE_PERIODIC_GRID_SEARCH = bool(controls.get("grid_search", False))
        return controls, version

    def _resize_stream_frame_for_processing(self, frame: np.ndarray) -> np.ndarray:
        if self.options.source_kind != "stream":
            return frame
        max_width = max(320, int(getattr(self.options, "stream_max_width", 960) or 960))
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / float(w)
        new_size = (max_width, max(1, int(round(h * scale))))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def _draw_candidates(self, display_frame: np.ndarray, candidates: List[dict], face_state_by_name: dict) -> None:
        h, w = display_frame.shape[:2]
        unknown_i = 0
        for candidate in candidates:
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

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(display_frame, (x1, max(0, y1 - 28)), (min(w - 1, x1 + 12 * len(label) + 20), y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 8, max(18, y1 - 8)), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)

    def _record_known_report(self, name: str, candidate: dict, attendance_marked: bool) -> None:
        report = self.known_report.setdefault(name, {
            "name": station.display_name_for_person(name),
            "detections": 0,
            "first_seen": now_iso(),
            "last_seen": now_iso(),
            "best_similarity": -1.0,
            "attendance_marks": 0,
        })
        report["detections"] += 1
        report["last_seen"] = now_iso()
        report["best_similarity"] = max(float(report.get("best_similarity", -1.0)), float(candidate.get("similarity", -1.0)))
        if attendance_marked:
            report["attendance_marks"] += 1

    def run(self) -> None:
        try:
            station.apply_algorithm_profile(self.options.profile or "fast_short", source="desktop")
            station.CAMERA_WIDTH = self.options.camera_width
            station.CAMERA_HEIGHT = self.options.camera_height
            station.ensure_model_file(station.YUNET_MODEL, station.YUNET_URL)
            station.ensure_model_file(station.SFACE_MODEL, station.SFACE_URL)
            station.check_opencv_api()

            detector, detector_backend = create_detector_with_backend(self.options.use_gpu)
            recognizer, recognizer_backend = create_recognizer_with_backend(self.options.use_gpu)
            backend_label = detector_backend if detector_backend == recognizer_backend else f"Detector: {detector_backend}; Recognizer: {recognizer_backend}"
            self._status("Loading known embeddings...", backend=backend_label)

            with self.known_features_lock:
                self.known_features = self._load_known_features(detector, recognizer)

            self.cap = self._open_capture()
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open video source: {self.options.source_value}")

            self._status("Station started", backend=backend_label, known_people=len(self.known_features))

            face_state_by_name = {}
            last_capture_time_by_name = {}
            saved_count_by_name = {}
            last_attendance_time_by_name = {}
            frame_index = 0
            last_candidates: List[dict] = []
            last_controls_version = -1
            last_fps_time = time.time()
            frames_since_fps = 0

            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.01)
                    continue

                frame = self._resize_stream_frame_for_processing(frame)

                _controls, controls_version = self._update_station_controls()
                if controls_version != last_controls_version:
                    # Old boxes were computed in the previous zoom/crop coordinate space.
                    last_candidates = []
                    last_controls_version = controls_version

                frame, _manual_zoom_crop = station.apply_manual_station_zoom(frame)
                zoom_status = station.manual_zoom_status_text()

                now = time.time()
                frame_index += 1
                frames_since_fps += 1
                if now - last_fps_time >= 1.0:
                    self.fps = frames_since_fps / max(0.001, now - last_fps_time)
                    frames_since_fps = 0
                    last_fps_time = now

                with self.known_features_lock:
                    known_features_copy = {k: list(v) for k, v in self.known_features.items()}

                if self.options.use_moodle:
                    known_features_copy = station.refresh_moodle_state_if_needed(known_features_copy)
                    with self.known_features_lock:
                        self.known_features = known_features_copy

                should_fast_search = frame_index % station.FAST_SEARCH_EVERY_N_FRAMES == 0
                should_full_grid_search = station.ENABLE_PERIODIC_GRID_SEARCH and frame_index % station.FULL_GRID_SEARCH_EVERY_N_FRAMES == 0
                candidates = last_candidates

                if should_fast_search or should_full_grid_search:
                    candidates = station.detect_faces_with_search_zoom(
                        detector=detector,
                        recognizer=recognizer,
                        known_features=known_features_copy,
                        frame=frame,
                        full_grid=should_full_grid_search,
                        last_box=None,
                        locked_name=None,
                    )
                    candidates = station.dedupe_candidates(candidates)
                    candidates = station.suppress_unknown_near_known(candidates)
                    last_candidates = candidates

                seen_known_names = set()
                for candidate in candidates:
                    if not candidate.get("is_known", False):
                        self.registry.update_from_candidate(candidate, frame)
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
                    })
                    state["stable_count"] += 1
                    state["last_seen"] = now
                    state["last_box"] = candidate.get("box")
                    state["last_similarity"] = candidate.get("similarity", -1.0)

                    attendance_marked = False
                    if state["stable_count"] >= station.STABLE_FRAMES_REQUIRED and station.SAVE_PHOTO_WHEN_KNOWN_STABLE:
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

                display_frame = frame.copy()
                self._draw_candidates(display_frame, candidates, face_state_by_name)

                known_count = sum(1 for c in candidates if c.get("is_known", False))
                unknown_count = len(candidates) - known_count
                status = f"FPS {self.fps:.1f} | faces {len(candidates)} | known {known_count} | unknown {unknown_count} | {backend_label}"
                if zoom_status != "zoom OFF":
                    status += f" | {zoom_status}"
                if self.options.source_kind == "stream":
                    status += f" | proc {display_frame.shape[1]}x{display_frame.shape[0]}"
                cv2.putText(display_frame, status, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                meta = {
                    "fps": self.fps,
                    "known_count": known_count,
                    "unknown_count": unknown_count,
                    "total_faces": len(candidates),
                    "backend": backend_label,
                    "zoom_status": zoom_status,
                    "known_report": list(self.known_report.values()),
                    "unknown_report": self.registry.snapshot(),
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
        self.geometry("1320x820")
        self.minsize(1120, 720)
        self.configure(bg="#101418")

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue = queue.Queue(maxsize=20)
        self.registry = UnknownRegistry()
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
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        camera_tab = ttk.Frame(self.notebook)
        reports_tab = ttk.Frame(self.notebook)
        self.notebook.add(camera_tab, text="Camera Station")
        self.notebook.add(reports_tab, text="Reports & Unknown Review")

        self._build_camera_tab(camera_tab)
        self._build_reports_tab(reports_tab)

    def _build_camera_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=0)
        parent.rowconfigure(0, weight=1)

        video_card = ttk.Frame(parent, style="Card.TFrame", padding=12)
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        video_card.rowconfigure(0, weight=1)
        video_card.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_card, text="Start the station to show the camera feed", anchor=tk.CENTER, style="Card.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        self.metrics_var = tk.StringVar(value="Faces: 0 | Known: 0 | Unknown: 0 | FPS: 0")
        ttk.Label(video_card, textvariable=self.metrics_var, style="Card.TLabel").grid(row=1, column=0, sticky="ew", pady=(10, 0))

        side = ttk.Frame(parent, style="Card.TFrame", padding=14)
        side.grid(row=0, column=1, sticky="ns")
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
        self.profile = tk.StringVar(value="fast_short")
        profile_box = ttk.Combobox(side, textvariable=self.profile, state="readonly", values=list(station.ALGORITHM_PROFILES.keys()), width=34)
        profile_box.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        self.use_gpu = tk.BooleanVar(value=True)
        self.use_moodle = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Try GPU acceleration", variable=self.use_gpu).grid(row=9, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Checkbutton(side, text="Load roster/embeddings from Moodle API", variable=self.use_moodle).grid(row=10, column=0, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Separator(side).grid(row=11, column=0, columnspan=3, sticky="ew", pady=14)
        ttk.Label(side, text="Digital zoom", style="Card.TLabel", font=("Segoe UI Semibold", 12)).grid(row=12, column=0, columnspan=3, sticky="w")

        self.zoom_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Enable zoom before recognition", variable=self.zoom_enabled, command=self.toggle_zoom).grid(row=13, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.zoom_factor = tk.DoubleVar(value=1.0)
        ttk.Label(side, text="Zoom factor", style="Card.TLabel").grid(row=14, column=0, sticky="w", pady=(10, 0))
        self.zoom_status_var = tk.StringVar(value="Zoom OFF")
        ttk.Label(side, textvariable=self.zoom_status_var, style="Card.TLabel").grid(row=14, column=1, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Scale(side, from_=1.0, to=8.0, variable=self.zoom_factor, command=lambda _v: self.on_zoom_factor_changed()).grid(row=15, column=0, columnspan=3, sticky="ew")

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

        self.draw_unknown = tk.BooleanVar(value=True)
        self.grid_search = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Draw unknown faces", variable=self.draw_unknown, command=self.push_controls).grid(row=21, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Checkbutton(side, text="Periodic grid search / slower high recall", variable=self.grid_search, command=self.push_controls).grid(row=22, column=0, columnspan=3, sticky="w", pady=(4, 0))

        controls = ttk.Frame(side, style="Card.TFrame")
        controls.grid(row=23, column=0, columnspan=3, sticky="ew", pady=(18, 0))
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.start_btn = ttk.Button(controls, text="Start", style="Primary.TButton", command=self.start_station)
        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop_station, state=tk.DISABLED)
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _build_reports_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)

        ttk.Label(parent, text="Known persons", style="Title.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(parent, text="Unknown review", style="Title.TLabel").grid(row=0, column=1, sticky="w", pady=(0, 8), padx=(12, 0))

        known_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        known_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        known_card.rowconfigure(0, weight=1)
        known_card.columnconfigure(0, weight=1)

        self.known_tree = ttk.Treeview(known_card, columns=("detections", "last_seen", "best", "marks"), show="tree headings")
        self.known_tree.heading("#0", text="Name")
        self.known_tree.heading("detections", text="Detections")
        self.known_tree.heading("last_seen", text="Last seen")
        self.known_tree.heading("best", text="Best sim")
        self.known_tree.heading("marks", text="Marks")
        self.known_tree.column("#0", width=220)
        self.known_tree.column("detections", width=90, anchor=tk.CENTER)
        self.known_tree.column("last_seen", width=150)
        self.known_tree.column("best", width=80, anchor=tk.CENTER)
        self.known_tree.column("marks", width=70, anchor=tk.CENTER)
        self.known_tree.grid(row=0, column=0, sticky="nsew")

        unknown_card = ttk.Frame(parent, style="Card.TFrame", padding=10)
        unknown_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        unknown_card.rowconfigure(0, weight=1)
        unknown_card.columnconfigure(0, weight=1)

        self.unknown_tree = ttk.Treeview(unknown_card, columns=("detections", "captures", "last_seen", "label"), show="tree headings")
        self.unknown_tree.heading("#0", text="Track")
        self.unknown_tree.heading("detections", text="Detections")
        self.unknown_tree.heading("captures", text="Pictures")
        self.unknown_tree.heading("last_seen", text="Last seen")
        self.unknown_tree.heading("label", text="Assigned label")
        self.unknown_tree.column("#0", width=105)
        self.unknown_tree.column("detections", width=85, anchor=tk.CENTER)
        self.unknown_tree.column("captures", width=75, anchor=tk.CENTER)
        self.unknown_tree.column("last_seen", width=150)
        self.unknown_tree.column("label", width=170)
        self.unknown_tree.grid(row=0, column=0, sticky="nsew")
        self.unknown_tree.bind("<<TreeviewSelect>>", self.on_unknown_selected)

        review_bar = ttk.Frame(unknown_card, style="Card.TFrame")
        review_bar.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        review_bar.columnconfigure(1, weight=1)
        ttk.Label(review_bar, text="Assign selected unknown as", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.assign_label_var = tk.StringVar()
        ttk.Entry(review_bar, textvariable=self.assign_label_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(review_bar, text="Assign label", command=self.assign_unknown_label).grid(row=0, column=2, sticky="ew")

        preview_frame = ttk.Frame(unknown_card, style="Card.TFrame")
        preview_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.preview_label = ttk.Label(preview_frame, text="Select an unknown track to preview the last captured picture.", style="Card.TLabel", anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(parent)
        buttons.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(buttons, text="Refresh report", command=self.refresh_reports).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Export CSV report", command=self.export_report_csv).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(buttons, text=f"Unknown pictures are saved under: {UNKNOWN_DIR}", style="Muted.TLabel").pack(side=tk.RIGHT)

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

        options = RuntimeOptions(
            source_kind=self.source_kind.get(),
            source_value=self.source_value.get(),
            embeddings_source=self.embeddings_source.get(),
            profile=self.profile.get(),
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

    def _effective_zoom_enabled(self) -> bool:
        return bool(self.zoom_enabled.get()) and float(self.zoom_factor.get()) > 1.001

    def _refresh_zoom_status_label(self) -> None:
        if not hasattr(self, "zoom_status_var"):
            return
        if self._effective_zoom_enabled():
            self.zoom_status_var.set(
                f"Zoom {float(self.zoom_factor.get()):.2f}x @ "
                f"{float(self.zoom_x.get()):.2f},{float(self.zoom_y.get()):.2f}"
            )
        else:
            self.zoom_status_var.set("Zoom OFF")

    def toggle_zoom(self) -> None:
        if self.zoom_enabled.get() and float(self.zoom_factor.get()) <= 1.001:
            # Checking the box should visibly change the image immediately.
            self.zoom_factor.set(2.0)
        self.push_controls()

    def on_zoom_factor_changed(self) -> None:
        # Moving the zoom slider above 1x automatically enables zoom.
        # Moving it back to 1x turns zoom off.
        self.zoom_enabled.set(float(self.zoom_factor.get()) > 1.001)
        self.push_controls()

    def push_controls(self) -> None:
        zoom_factor = float(self.zoom_factor.get())
        zoom_enabled = bool(self.zoom_enabled.get()) and zoom_factor > 1.001
        self._refresh_zoom_status_label()
        if self.worker:
            self.worker.update_controls(
                zoom_enabled=zoom_enabled,
                zoom_factor=zoom_factor,
                zoom_center_x=float(self.zoom_x.get()),
                zoom_center_y=float(self.zoom_y.get()),
                draw_unknown=bool(self.draw_unknown.get()),
                grid_search=bool(self.grid_search.get()),
            )

    def pan_zoom(self, dx: float, dy: float) -> None:
        if float(self.zoom_factor.get()) <= 1.001:
            self.zoom_factor.set(2.0)
        self.zoom_enabled.set(True)
        # Fixed pan step. This moves the crop center enough to be visible on network streams.
        self.zoom_x.set(min(1.0, max(0.0, self.zoom_x.get() + dx)))
        self.zoom_y.set(min(1.0, max(0.0, self.zoom_y.get() + dy)))
        self.push_controls()

    def reset_zoom(self) -> None:
        self.zoom_enabled.set(False)
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
                f"Faces: {meta.get('total_faces', 0)} | Known: {meta.get('known_count', 0)} | "
                f"Unknown: {meta.get('unknown_count', 0)} | FPS: {meta.get('fps', 0.0):.1f}"
            )
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
        max_w = max(640, self.video_label.winfo_width() or 960)
        max_h = max(360, self.video_label.winfo_height() or 540)
        image.thumbnail((max_w, max_h), Image.LANCZOS)
        self.last_photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.last_photo, text="")

    def refresh_reports(self, meta: Optional[dict] = None) -> None:
        meta = meta or self.latest_meta or {}
        known_rows = meta.get("known_report", [])
        unknown_rows = meta.get("unknown_report", self.registry.snapshot())

        known_existing = set(self.known_tree.get_children())
        for row in known_rows:
            iid = safe_slug(row.get("name"), "known")
            values = (
                int(row.get("detections", 0)),
                row.get("last_seen", ""),
                f"{float(row.get('best_similarity', -1.0)):.3f}" if row.get("best_similarity", -1.0) is not None else "",
                int(row.get("attendance_marks", 0)),
            )
            if iid in known_existing:
                self.known_tree.item(iid, text=row.get("name", iid), values=values)
                known_existing.remove(iid)
            else:
                self.known_tree.insert("", tk.END, iid=iid, text=row.get("name", iid), values=values)
        for iid in known_existing:
            self.known_tree.delete(iid)

        unknown_existing = set(self.unknown_tree.get_children())
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
                unknown_existing.remove(iid)
            else:
                self.unknown_tree.insert("", tk.END, iid=iid, text=iid, values=values)
        for iid in unknown_existing:
            self.unknown_tree.delete(iid)

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
        unknown_rows = meta.get("unknown_report", self.registry.snapshot())
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
