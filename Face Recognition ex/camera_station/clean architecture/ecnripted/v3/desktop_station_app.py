import csv
import json
import math
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import zipfile
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

# When the GUI imports the station module, the module-level __main__ block from
# moodle_yunet_sface_station.py is not executed. Therefore the GUI must explicitly
# load station_config.json; otherwise Moodle API defaults are used and the HMAC
# secret stays empty/default.
try:
    station.apply_runtime_options()
except SystemExit:
    raise
except Exception as _station_config_exc:
    print(f"[CONFIG] Station config error: {_station_config_exc}")
    raise

try:
    from faceattendance_clean.application.use_cases import VideoPlaybackController, parse_speed_choice
    from faceattendance_clean.infrastructure.legacy_station_adapter import LegacyStationModuleAdapter
    from faceattendance_clean.presentation.theme import MODERN_PALETTE, apply_modern_theme
    from faceattendance_clean.presentation.widgets import configure_tree_tags, metric_card, retag_tree_rows, section_title
except Exception as _clean_import_error:  # pragma: no cover - keep legacy startup robust
    VideoPlaybackController = None
    LegacyStationModuleAdapter = None
    MODERN_PALETTE = None
    apply_modern_theme = None
    configure_tree_tags = None
    metric_card = None
    retag_tree_rows = None
    section_title = None
    print(f"[CLEAN] Optional clean architecture helpers disabled: {_clean_import_error}")

APP_DIR = Path(__file__).resolve().parent
UNKNOWN_DIR = APP_DIR / "unknown_review"
REVIEWED_EMBEDDINGS_DIR = APP_DIR / "reviewed_embeddings"
DELETED_KNOWN_BACKUP_DIR = APP_DIR / "deleted_known_faces_backups"
REPORTS_DIR = APP_DIR / "reports"


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_slug(text: str, fallback: str = "person") -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "").strip()).strip("_")
    return slug or fallback


def is_path_inside(child: Path, parent: Path) -> bool:
    """Return True when child is inside parent, compatible with older Python versions."""
    try:
        child_resolved = Path(child).resolve()
        parent_resolved = Path(parent).resolve()
        child_resolved.relative_to(parent_resolved)
        return True
    except Exception:
        return False


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


def merge_known_feature_sets(target: Dict[str, List[np.ndarray]], extra: Dict[str, List[np.ndarray]]) -> int:
    """Merge normalized SFace descriptors into the active known-face database."""
    added = 0
    for label, descriptors in (extra or {}).items():
        label = str(label or "").strip()
        if not label:
            continue
        bucket = target.setdefault(label, [])
        for descriptor in descriptors or []:
            desc = normalized_descriptor(descriptor)
            if desc is None:
                continue
            bucket.append(desc.astype(np.float32))
            added += 1
    return added


def load_reviewed_known_features() -> Dict[str, List[np.ndarray]]:
    """Load labels created from Unknown review assignments.

    These files are intentionally stored outside the original enrollment folder,
    therefore every station session must merge them explicitly.  Without this,
    a label assigned in Unknown review remains visible in the list but the person
    is not actually part of the recognition database after restart.
    """
    if not REVIEWED_EMBEDDINGS_DIR.exists():
        return {}
    try:
        files = station.find_embedding_files(REVIEWED_EMBEDDINGS_DIR) if hasattr(station, "find_embedding_files") else list(REVIEWED_EMBEDDINGS_DIR.rglob("*.json"))
    except Exception:
        files = list(REVIEWED_EMBEDDINGS_DIR.rglob("*.json"))
    if not files:
        return {}
    try:
        return station.load_known_features_from_embeddings(REVIEWED_EMBEDDINGS_DIR)
    except Exception as exc:
        print(f"[WARN] Could not load reviewed unknown embeddings from {REVIEWED_EMBEDDINGS_DIR}: {exc}")
        return {}


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
            # Save the first usable crop immediately.  The walk-through profile can
            # hide repeated unknown faces after only a few stable detections; waiting
            # until detection 10 meant that unknown tracks were drawn as UNK on the
            # live frame but never received a review picture, so they could not be
            # inspected and labelled manually.
            should_capture = created_new_track or (
                detection_no % self.capture_every == 0 and elapsed >= self.capture_min_interval_seconds
            )
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

    def clear_assigned_labels(self) -> int:
        """Make all reviewed/labeled unknown tracks unassigned again.

        This is used by the destructive 'delete all known faces' button so no
        old Unknown-review label remains visible after its embedding file was
        deleted.  The captured face pictures are kept for review/re-labeling.
        """
        changed = 0
        with self.lock:
            for track in self.tracks.values():
                if str(track.get("assigned_label") or "").strip():
                    track["assigned_label"] = ""
                    track.pop("assigned_at", None)
                    track.pop("reviewed_embedding_file", None)
                    changed += 1
            if changed:
                self.save(force=True)
        return changed


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

    @staticmethod
    def _safe_zip_name(name: str) -> Optional[str]:
        value = str(name or "").replace("\\", "/").strip("/")
        if not value or value.startswith("/") or ".." in Path(value).parts:
            return None
        return value

    @staticmethod
    def _unique_file_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for index in range(1, 10000):
            candidate = parent / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not find a free file name for {path}")

    def export_package(self, output_zip_path: Path, include_reviewed_embeddings: bool = True) -> Tuple[int, int, int, int, int]:
        """Export the full review database: unlabeled unknowns and already labeled faces.

        The canonical importable data stays under unknown_review/ and reviewed_embeddings/.
        For convenience, the same images that belong to assigned/labeled tracks are
        also copied under labeled_faces/<label>/ so the ZIP can be opened and used
        directly as a small labeled image dataset.
        """
        output_zip_path = Path(output_zip_path)
        output_zip_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            self.save(force=True)
            track_rows = [dict(track) for track in self.tracks.values()]
            tracks_count = len(track_rows)

        image_count = 0
        reviewed_count = 0
        labeled_tracks_count = sum(1 for track in track_rows if str(track.get("assigned_label") or "").strip())
        labeled_image_count = 0
        manifest = {
            "version": 2,
            "created_at": now_iso(),
            "type": "faceattendance_review_export",
            "contains": [
                "unknown_review/unknown_registry.json",
                "unknown_review/images",
                "reviewed_embeddings",
                "labeled_faces",
            ],
            "description": "Full Unknown review export: unlabeled unknown tracks plus already labeled/reviewed faces and their images.",
        }

        with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            if self.db_path.exists():
                zf.write(self.db_path, "unknown_review/unknown_registry.json")
            if self.images_dir.exists():
                for image_path in sorted(self.images_dir.rglob("*")):
                    if not image_path.is_file():
                        continue
                    zf.write(image_path, f"unknown_review/images/{image_path.relative_to(self.images_dir).as_posix()}")
                    image_count += 1

            labeled_export_names = set()
            for track in track_rows:
                label = str(track.get("assigned_label") or "").strip()
                if not label:
                    continue
                label_dir = safe_slug(label, "labeled_person")
                for capture in track.get("captures", []) or []:
                    if not isinstance(capture, dict):
                        continue
                    rel = str(capture.get("path") or "").replace("\\", "/").lstrip("/")
                    if not rel:
                        continue
                    image_path = self.base_dir / rel
                    if not image_path.exists() or not image_path.is_file():
                        continue
                    arc_name = f"labeled_faces/{label_dir}/{track.get('id', 'UNK')}_{image_path.name}"
                    if arc_name in labeled_export_names:
                        continue
                    zf.write(image_path, arc_name)
                    labeled_export_names.add(arc_name)
                    labeled_image_count += 1

            if include_reviewed_embeddings and REVIEWED_EMBEDDINGS_DIR.exists():
                for emb_path in sorted(REVIEWED_EMBEDDINGS_DIR.rglob("*")):
                    if not emb_path.is_file():
                        continue
                    zf.write(emb_path, f"reviewed_embeddings/{emb_path.relative_to(REVIEWED_EMBEDDINGS_DIR).as_posix()}")
                    reviewed_count += 1

        return tracks_count, image_count, reviewed_count, labeled_tracks_count, labeled_image_count

    def _allocate_import_track_id(self, requested_id: str) -> str:
        requested_id = str(requested_id or "").strip()
        if requested_id and requested_id not in self.tracks:
            m = re.search(r"(\d+)$", requested_id)
            if m:
                self.next_number = max(self.next_number, int(m.group(1)) + 1)
            return requested_id

        while True:
            candidate = f"UNK-{self.next_number:04d}"
            self.next_number += 1
            if candidate not in self.tracks:
                return candidate

    def import_package(self, input_zip_path: Path) -> Tuple[int, int, int]:
        """Merge an exported unknown-review package into the local registry."""
        input_zip_path = Path(input_zip_path)
        if not input_zip_path.exists():
            raise FileNotFoundError(str(input_zip_path))

        imported_tracks = 0
        imported_images = 0
        imported_reviewed = 0
        capture_path_map: Dict[str, str] = {}

        with self.lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)
            REVIEWED_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(input_zip_path, "r") as zf:
                members = list(zf.infolist())
                for info in members:
                    safe_name = self._safe_zip_name(info.filename)
                    if not safe_name or info.is_dir():
                        continue
                    if not (
                        safe_name == "unknown_review/unknown_registry.json"
                        or safe_name.startswith("unknown_review/images/")
                        or safe_name.startswith("reviewed_embeddings/")
                        or safe_name == "manifest.json"
                    ):
                        continue

                    if safe_name.startswith("unknown_review/images/"):
                        relative_inside_images = safe_name[len("unknown_review/images/"):]
                        source_rel = f"images/{relative_inside_images}"
                        target_path = self._unique_file_path(self.images_dir / Path(relative_inside_images).name)
                        with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        capture_path_map[source_rel.replace("\\", "/")] = f"images/{target_path.name}"
                        imported_images += 1

                    elif safe_name.startswith("reviewed_embeddings/"):
                        relative_inside_reviewed = safe_name[len("reviewed_embeddings/"):]
                        target_path = self._unique_file_path(REVIEWED_EMBEDDINGS_DIR / Path(relative_inside_reviewed).name)
                        with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        imported_reviewed += 1

                registry_member = None
                for info in members:
                    if self._safe_zip_name(info.filename) == "unknown_review/unknown_registry.json":
                        registry_member = info
                        break

                if registry_member is not None:
                    with zf.open(registry_member, "r") as f:
                        imported_data = json.loads(f.read().decode("utf-8"))
                    for raw_track in imported_data.get("tracks", []) or []:
                        if not isinstance(raw_track, dict):
                            continue
                        track = dict(raw_track)
                        track["id"] = self._allocate_import_track_id(str(track.get("id") or ""))
                        captures = []
                        for raw_capture in track.get("captures", []) or []:
                            if not isinstance(raw_capture, dict):
                                continue
                            capture = dict(raw_capture)
                            rel = str(capture.get("path") or "").replace("\\", "/").lstrip("/")
                            if rel in capture_path_map:
                                capture["path"] = capture_path_map[rel]
                            elif rel and not (self.base_dir / rel).exists():
                                # Keep metadata, but do not point to a non-existing image.
                                capture["missing_imported_image"] = rel
                                capture["path"] = ""
                            captures.append(capture)
                        track["captures"] = captures
                        track.setdefault("imported_at", now_iso())
                        self.tracks[track["id"]] = track
                        imported_tracks += 1

            self.save(force=True)

        return imported_tracks, imported_images, imported_reviewed

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


def _is_url_source(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", str(value or "").strip()))


def _is_rtsp_source(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("rtsp://", "rtsps://"))


def _is_http_stream_source(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("http://", "https://"))


def _build_rtsp_ffmpeg_capture_options(transport: str) -> str:
    """Return OpenCV FFmpeg options for RTSP capture.

    OpenCV's Python wheel usually uses FFmpeg internally for RTSP.  The most
    reliable fix for NVR/IP-camera streams on Windows is forcing RTSP over TCP
    and keeping the input buffer small, otherwise `cap.read()` often freezes,
    lags several seconds behind, or fails after a few frames.
    """
    transport = str(transport or "tcp").strip().lower()
    if transport not in {"tcp", "udp"}:
        transport = "tcp"

    # Values are FFmpeg/OpenCV capture options in key;value form.  Timeout
    # values are microseconds.  Keep this list conservative because older
    # OpenCV wheels may reject unusual AVOptions.
    options = [
        ("rtsp_transport", transport),
        ("stimeout", "5000000"),
        ("rw_timeout", "5000000"),
        ("fflags", "nobuffer"),
        ("flags", "low_delay"),
        ("max_delay", "500000"),
    ]
    return "|".join(f"{key};{value}" for key, value in options)


@dataclass
class RuntimeOptions:
    source_kind: str
    source_value: str
    embeddings_source: str
    profile: str
    use_gpu: bool
    use_moodle: bool
    # For RTSP streams, TCP is much more reliable on Windows/NVR setups.
    rtsp_transport: str = "tcp"
    # 0 means: use the selected algorithm profile resolution.
    camera_width: int = 0
    camera_height: int = 0
    # Video-file playback controls. They are ignored for live cameras/streams.
    playback_speed: float = 1.0
    # When true, no known embeddings are loaded or refreshed; every face stays unknown.
    known_database_disabled: bool = False


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
            "known_database_disabled": bool(getattr(options, "known_database_disabled", False)),
            "playback_paused": False,
            "playback_speed": max(0.10, min(8.0, float(getattr(options, "playback_speed", 1.0) or 1.0))),
            # Video seeking is request/token based so the worker performs each
            # forward/back action exactly once, even if push_controls() runs often.
            "seek_token": 0,
            "seek_delta_seconds": 0.0,
            "seek_absolute_frame": None,
        }
        self.controls_version = 0
        self.known_features_lock = threading.RLock()
        self.known_features: Dict[str, List[np.ndarray]] = {}
        self.known_feature_index = station.build_known_feature_index(self.known_features)
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
                self.controls_version = getattr(self, "controls_version", 0) + 1

    def register_known(self, label: str, descriptors: List[np.ndarray]) -> None:
        # A new manual label/import should re-enable recognition after a previous
        # "delete all known faces" action.
        self.update_controls(known_database_disabled=False)
        with self.known_features_lock:
            for descriptor in descriptors:
                desc = normalized_descriptor(descriptor)
                if desc is not None:
                    self.known_features.setdefault(label, []).append(desc.astype(np.float32))
            self.known_feature_index = station.build_known_feature_index(self.known_features)

    def clear_known_database(self, disable_reload: bool = True) -> None:
        """Clear all known descriptors and known-person report rows in this worker."""
        if disable_reload:
            self.update_controls(known_database_disabled=True)
        with self.known_features_lock:
            self.known_features = {}
            self.known_feature_index = station.build_known_feature_index(self.known_features)
            self.known_report = {}

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

    def _is_video_file_source(self) -> bool:
        source_kind = str(self.options.source_kind or "")
        source_value = str(self.options.source_value or "").strip()
        if source_kind == "video_file":
            return True
        if source_kind != "stream" or not source_value:
            return False
        # Treat plain local paths as video-file playback.  URLs such as RTSP/HTTP
        # remain live streams and are not paced or paused like files.
        if _is_url_source(source_value):
            return False
        try:
            return Path(source_value).exists()
        except Exception:
            return False

    def _is_live_stream_source(self) -> bool:
        source_kind = str(self.options.source_kind or "")
        source_value = str(self.options.source_value or "").strip()
        return source_kind == "stream" and (_is_rtsp_source(source_value) or _is_http_stream_source(source_value))

    def _sleep_interruptibly(self, seconds: float) -> None:
        end_time = time.perf_counter() + max(0.0, float(seconds))
        while not self.stop_event.is_set():
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                return
            self.stop_event.wait(min(0.05, remaining))

    def _pace_video_playback(self, loop_started: float, source_fps: float, playback_speed: float) -> None:
        if source_fps <= 0:
            source_fps = 25.0
        speed = max(0.10, min(8.0, float(playback_speed or 1.0)))
        target_delay = 1.0 / max(1.0, source_fps * speed)
        elapsed = time.perf_counter() - loop_started
        if target_delay > elapsed:
            self._sleep_interruptibly(target_delay - elapsed)

    @staticmethod
    def _format_seek_delta(seconds: float) -> str:
        sign = "+" if seconds >= 0 else "-"
        total_seconds = abs(int(round(float(seconds))))
        minutes, sec = divmod(total_seconds, 60)
        if minutes:
            return f"{sign}{minutes}m{sec:02d}s"
        return f"{sign}{sec}s"

    def _apply_video_seek_request(
        self,
        controls: dict,
        last_seek_token: int,
        source_fps: float,
        video_total_frames: int,
    ) -> Tuple[bool, int]:
        """Apply a pending video seek request and return (applied, token).

        The GUI can call update_controls() many times per second.  A monotonically
        increasing seek token makes each seek command idempotent inside the worker.
        """
        try:
            token = int(controls.get("seek_token", 0) or 0)
        except Exception:
            token = 0
        if token <= 0 or token == last_seek_token or self.cap is None:
            return False, last_seek_token

        fps = max(1.0, float(source_fps or 25.0))
        try:
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        except Exception:
            current_frame = 0

        absolute_frame = controls.get("seek_absolute_frame", None)
        if absolute_frame is not None:
            try:
                target_frame = int(round(float(absolute_frame)))
            except Exception:
                target_frame = current_frame
            seek_label = f"frame {target_frame}"
        else:
            try:
                delta_seconds = float(controls.get("seek_delta_seconds", 0.0) or 0.0)
            except Exception:
                delta_seconds = 0.0
            target_frame = current_frame + int(round(delta_seconds * fps))
            seek_label = self._format_seek_delta(delta_seconds)

        if video_total_frames > 0:
            target_frame = max(0, min(max(0, video_total_frames - 1), target_frame))
        else:
            target_frame = max(0, target_frame)

        applied = False
        try:
            applied = bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame))
        except Exception:
            applied = False

        if not applied:
            # Some backends are more reliable when seeking by timestamp.
            try:
                applied = bool(self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * target_frame / fps))
            except Exception:
                applied = False

        if applied:
            self._status(f"Video seek {seek_label} -> frame {target_frame}")
        else:
            self._status(f"Video seek requested, but this file/backend did not accept seeking")
        return True, token

    def _try_open_ffmpeg_capture(self, source_value: str, capture_options: Optional[str] = None):
        previous_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        if capture_options:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = capture_options

        params = []
        if _is_rtsp_source(source_value):
            for prop_name in ("CAP_PROP_OPEN_TIMEOUT_MSEC", "CAP_PROP_READ_TIMEOUT_MSEC"):
                prop = getattr(cv2, prop_name, None)
                if prop is not None:
                    params.extend([int(prop), 5000])

        try:
            if params:
                try:
                    cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG, params)
                except Exception:
                    cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
        finally:
            if capture_options:
                if previous_options is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous_options
        return cap

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

        # RTSP/HTTP stream or local video file.  RTSP is opened with FFmpeg
        # options first because this is the most stable path for Windows + NVR
        # cameras.  If the selected transport fails, fall back to TCP, then to
        # OpenCV's default backend.
        is_rtsp = _is_rtsp_source(source_value)
        cap = None
        if is_rtsp:
            selected_transport = str(getattr(self.options, "rtsp_transport", "tcp") or "tcp").strip().lower()
            transports = [selected_transport]
            if selected_transport != "tcp":
                transports.append("tcp")
            if selected_transport != "udp":
                transports.append("udp")

            for transport in transports:
                options = _build_rtsp_ffmpeg_capture_options(transport)
                cap = self._try_open_ffmpeg_capture(source_value, options)
                if cap is not None and cap.isOpened():
                    self._status(f"RTSP stream opened with FFmpeg/{transport.upper()}")
                    break
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                cap = None

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = self._try_open_ffmpeg_capture(source_value)

        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source_value)

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _reopen_capture_after_stream_failure(self) -> bool:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self._sleep_interruptibly(0.25)
        self.cap = self._open_capture()
        return bool(self.cap is not None and self.cap.isOpened())

    def _load_known_features(self, detector, recognizer) -> Dict[str, List[np.ndarray]]:
        with self.controls_lock:
            if bool(self.controls.get("known_database_disabled", False)):
                return {}
        if self.options.use_moodle:
            station.USE_MOODLE_API = True
            known_features = station.load_known_features_from_moodle()
        else:
            station.USE_MOODLE_API = False
            station.EMBEDDINGS_SOURCE = self.options.embeddings_source or station.EMBEDDINGS_SOURCE
            known_features = station.load_known_features_from_embeddings(station.EMBEDDINGS_SOURCE)

        reviewed_features = load_reviewed_known_features()
        reviewed_count = merge_known_feature_sets(known_features, reviewed_features)
        if reviewed_count:
            print(f"[OK] Loaded {reviewed_count} reviewed unknown embedding(s) from {REVIEWED_EMBEDDINGS_DIR}")
        return known_features

    def _update_station_controls(self) -> Tuple[dict, int]:
        with self.controls_lock:
            controls = dict(self.controls)
            version = self.controls_version

        zoom_factor = max(1.0, float(controls.get("zoom_factor", 1.0)))
        zoom_enabled = bool(controls.get("zoom_enabled")) and zoom_factor > 1.001

        # The desktop UI scale goes to 8x. Keep the station-side clamp in sync;
        # otherwise moving the GUI above the old default max silently stops at 5x.
        station.MANUAL_ZOOM_MAX = max(float(getattr(station, "MANUAL_ZOOM_MAX", 5.0)), 8.0)
        station.MANUAL_ZOOM_ENABLED = zoom_enabled
        station.MANUAL_ZOOM_FACTOR = zoom_factor
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
        return controls, version

    def _draw_candidates(self, display_frame: np.ndarray, candidates: List[dict], face_state_by_name: dict, unknown_state_by_id: dict) -> None:
        h, w = display_frame.shape[:2]
        unknown_i = 0
        for candidate in candidates:
            if station.is_candidate_resolved_hidden(candidate, face_state_by_name, unknown_state_by_id):
                continue

            is_known = bool(candidate.get("is_known"))
            is_accepted_known = station.is_candidate_accepted_known(candidate)
            is_pending_identity = station.is_candidate_pending_identity(candidate)
            if not is_known and not station.DRAW_UNKNOWN_FACES:
                continue

            x1, y1, x2, y2 = [int(v) for v in candidate.get("box", [0, 0, 0, 0])]
            x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
            y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
            color = (35, 190, 90) if is_accepted_known else ((35, 170, 220) if is_pending_identity else (45, 70, 230))

            if is_accepted_known:
                raw_name = str(candidate.get("name", ""))
                label = station.short_display_name(station.display_name_for_person(raw_name))
                stable = face_state_by_name.get(raw_name, {}).get("stable_count", 0)
                label = f"{label} {min(stable, station.STABLE_FRAMES_REQUIRED)}/{station.STABLE_FRAMES_REQUIRED}"
            elif is_pending_identity:
                raw_name = str(candidate.get("raw_name") or candidate.get("name") or "")
                label_name = station.short_display_name(station.display_name_for_person(raw_name))
                votes = int(candidate.get("identity_vote_count", 0) or 0)
                need = int(getattr(station, "IDENTITY_MIN_KNOWN_VOTES", 4) or 4)
                label = f"CHECK {label_name} {min(votes, need)}/{need}"
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

            is_video_playback = self._is_video_file_source()
            is_live_stream = self._is_live_stream_source()
            try:
                source_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception:
                source_fps = 0.0
            if not math.isfinite(source_fps) or source_fps <= 1.0:
                source_fps = 25.0
            try:
                video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if is_video_playback else 0
            except Exception:
                video_total_frames = 0

            if is_video_playback:
                self._status(
                    f"Video recognition started ({source_fps:.1f} FPS source)",
                    backend=backend_label,
                    known_people=len(self.known_features),
                )
            else:
                self._status("Station started", backend=backend_label, known_people=len(self.known_features))

            face_state_by_name = {}
            unknown_state_by_id = {"_next_number": 1}
            identity_state_by_track = {"_next_number": 1}
            last_capture_time_by_name = {}
            saved_count_by_name = {}
            last_attendance_time_by_name = {}
            frame_index = 0
            last_candidates: List[dict] = []
            last_controls_version = -1
            last_fps_time = time.time()
            frames_since_fps = 0
            last_report_snapshot_time = 0.0
            consecutive_read_failures = 0
            last_stream_reconnect_time = 0.0
            last_video_seek_token = 0
            force_detection_after_seek = False

            while not self.stop_event.is_set():
                loop_started = time.perf_counter()
                _controls, controls_version = self._update_station_controls()
                if controls_version != last_controls_version:
                    last_candidates = []
                    last_controls_version = controls_version

                seek_applied = False
                if is_video_playback:
                    seek_applied, last_video_seek_token = self._apply_video_seek_request(
                        _controls,
                        last_video_seek_token,
                        source_fps,
                        video_total_frames,
                    )
                    if seek_applied:
                        last_candidates = []
                        force_detection_after_seek = True

                    while (
                        not self.stop_event.is_set()
                        and bool(_controls.get("playback_paused", False))
                        and not seek_applied
                    ):
                        self._sleep_interruptibly(0.05)
                        _controls, controls_version = self._update_station_controls()
                        if controls_version != last_controls_version:
                            last_candidates = []
                            last_controls_version = controls_version
                        seek_applied, last_video_seek_token = self._apply_video_seek_request(
                            _controls,
                            last_video_seek_token,
                            source_fps,
                            video_total_frames,
                        )
                        if seek_applied:
                            last_candidates = []
                            force_detection_after_seek = True
                            break
                    if self.stop_event.is_set():
                        break

                ok, frame = self.cap.read()
                if not ok or frame is None:
                    if is_video_playback:
                        self._status("Video finished", backend=backend_label)
                        break

                    consecutive_read_failures += 1
                    if is_live_stream:
                        now_perf = time.perf_counter()
                        if consecutive_read_failures == 1:
                            self._status("Waiting for RTSP/HTTP frames...", backend=backend_label)
                        if consecutive_read_failures >= 12 and (now_perf - last_stream_reconnect_time) >= 1.5:
                            self._status("RTSP/HTTP stream stalled; reconnecting...", backend=backend_label)
                            last_stream_reconnect_time = now_perf
                            if self._reopen_capture_after_stream_failure():
                                consecutive_read_failures = 0
                                self._status("RTSP/HTTP stream reconnected", backend=backend_label)
                            else:
                                self._status("RTSP/HTTP reconnect failed; retrying...", backend=backend_label)
                                consecutive_read_failures = 0
                                self._sleep_interruptibly(0.75)
                        else:
                            self._sleep_interruptibly(0.04)
                    else:
                        self._sleep_interruptibly(0.04)
                    continue

                consecutive_read_failures = 0

                frame, _manual_zoom_crop = station.apply_manual_station_zoom(frame)
                zoom_status = station.manual_zoom_status_text()

                now = time.time()
                frame_index += 1
                frames_since_fps += 1
                if now - last_fps_time >= 1.0:
                    self.fps = frames_since_fps / max(0.001, now - last_fps_time)
                    frames_since_fps = 0
                    last_fps_time = now

                known_db_disabled = bool(_controls.get("known_database_disabled", False))
                with self.known_features_lock:
                    if known_db_disabled and (self.known_features or self.known_feature_index.get("names")):
                        self.known_features = {}
                        self.known_feature_index = station.build_known_feature_index(self.known_features)
                        self.known_report = {}
                    known_features_index = self.known_feature_index

                if self.options.use_moodle and not known_db_disabled:
                    refreshed = station.refresh_moodle_state_if_needed(self.known_features)
                    if refreshed is not self.known_features:
                        # Moodle roster refresh returns the server-side embeddings only.
                        # Re-merge the local Unknown-review labels so they are not lost
                        # a few minutes after assignment.
                        merge_known_feature_sets(refreshed, load_reviewed_known_features())
                        with self.known_features_lock:
                            self.known_features = refreshed
                            self.known_feature_index = station.build_known_feature_index(self.known_features)
                            known_features_index = self.known_feature_index

                should_fast_search = frame_index % station.FAST_SEARCH_EVERY_N_FRAMES == 0
                should_full_grid_search = station.ENABLE_PERIODIC_GRID_SEARCH and frame_index % station.FULL_GRID_SEARCH_EVERY_N_FRAMES == 0
                candidates = last_candidates

                if force_detection_after_seek or should_fast_search or should_full_grid_search:
                    force_detection_after_seek = False
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
                    station.update_identity_vote_tracking(candidates, identity_state_by_track, now)
                    last_candidates = candidates

                track_t0 = time.perf_counter()
                # Only truly unknown faces go to Unknown tracking/review. Known labels
                # that are still in the temporal voting stage must not pollute either
                # Known persons or Unknown review.
                review_unknown_candidates = [
                    c for c in candidates
                    if not c.get("is_known", False) and not station.is_candidate_pending_identity(c)
                ]
                station.update_unknown_tracking(review_unknown_candidates, unknown_state_by_id, now)

                seen_known_names = set()
                unknown_registry_updates = 0
                unknown_registry_budget = max(0, int(getattr(station, "UNKNOWN_REGISTRY_MAX_UPDATES_PER_FRAME", 2)))
                for candidate in candidates:
                    if station.is_candidate_pending_identity(candidate):
                        continue

                    if not candidate.get("is_known", False):
                        if (
                            unknown_registry_budget != 0
                            and unknown_registry_updates < unknown_registry_budget
                            and not station.is_candidate_confirmed_unknown_hidden(candidate, unknown_state_by_id)
                        ):
                            self.registry.update_from_candidate(candidate, frame)
                            unknown_registry_updates += 1
                        continue

                    if not station.is_candidate_accepted_known(candidate):
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
                # identity vote tracks are cleaned inside update_identity_vote_tracking
                track_ms = (time.perf_counter() - track_t0) * 1000.0

                should_output_frame = (frame_index % max(1, int(getattr(station, "DESKTOP_OUTPUT_EVERY_N_FRAMES", 1))) == 0)
                if not should_output_frame:
                    if is_video_playback:
                        self._pace_video_playback(loop_started, source_fps, float(_controls.get("playback_speed", 1.0)))
                    continue

                draw_t0 = time.perf_counter()
                display_frame = frame.copy()
                self._draw_candidates(display_frame, candidates, face_state_by_name, unknown_state_by_id)
                draw_ms = (time.perf_counter() - draw_t0) * 1000.0

                visible_candidates, hidden_confirmed_candidates = station.split_candidates_by_confirmation(candidates, face_state_by_name, unknown_state_by_id)
                known_count = sum(1 for c in visible_candidates if station.is_candidate_accepted_known(c))
                pending_count = sum(1 for c in visible_candidates if station.is_candidate_pending_identity(c))
                unknown_count = len(visible_candidates) - known_count - pending_count
                hidden_confirmed_count = len(hidden_confirmed_candidates)
                status = (
                    f"FPS {self.fps:.1f} | faces {len(candidates)} | remaining {len(visible_candidates)}"
                    f" | known {known_count} | unknown {unknown_count}"
                )
                if pending_count:
                    status += f" | checking {pending_count}"
                if hidden_confirmed_count:
                    status += f" | resolved hidden {hidden_confirmed_count}"
                status += f" | SFace {getattr(station, 'LAST_SFACE_CALLS', 0)}/frame"
                batches = int(getattr(station, 'LAST_SFACE_BATCHES', 0) or 0)
                if batches:
                    status += f"/{batches} batch"
                status += f" | {backend_label}"
                if bool(_controls.get("known_database_disabled", False)):
                    status += " | known DB disabled"
                if zoom_status != "zoom OFF":
                    status += f" | {zoom_status}"
                cv2.putText(display_frame, status, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                meta = {
                    "fps": self.fps,
                    "known_count": known_count,
                    "unknown_count": unknown_count,
                    "pending_identity_count": pending_count,
                    "total_faces": len(candidates),
                    "remaining_faces": len(visible_candidates),
                    "hidden_confirmed_count": hidden_confirmed_count,
                    "backend": backend_label,
                    "zoom_status": zoom_status,
                    "known_report": self._normalized_known_report_rows(),
                    "unknown_report": self.registry.snapshot(),
                }
                if is_video_playback:
                    speed = max(0.10, min(8.0, float(_controls.get("playback_speed", 1.0) or 1.0)))
                    try:
                        position_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                    except Exception:
                        position_frame = frame_index
                    try:
                        position_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
                    except Exception:
                        position_ms = 0.0
                    meta.update({
                        "is_video_playback": True,
                        "video_position_frame": position_frame,
                        "video_total_frames": video_total_frames,
                        "video_position_ms": position_ms,
                        "video_fps": source_fps,
                        "playback_speed": speed,
                        "playback_paused": bool(_controls.get("playback_paused", False)),
                    })
                self._put_frame(display_frame, meta)
                if is_video_playback:
                    self._pace_video_playback(loop_started, source_fps, float(_controls.get("playback_speed", 1.0)))

        except Exception as exc:
            self._status(f"ERROR: {exc}", error=traceback.format_exc())
        finally:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self._status("Station stopped")


class VideoRecognitionWindow(tk.Toplevel):
    """Separate window for running the same FR pipeline over a local video file."""

    SPEED_MIN = 0.25
    SPEED_MAX = 4.0

    def __init__(self, parent: "DesktopStationApp"):
        super().__init__(parent)
        self.parent_app = parent
        self.title("Video File Face Recognition")
        self.geometry("1180x760")
        self.minsize(900, 600)
        palette_bg = getattr(MODERN_PALETTE, "bg", "#101418") if MODERN_PALETTE is not None else "#101418"
        self.configure(bg=palette_bg)
        if apply_modern_theme is not None:
            try:
                apply_modern_theme(self)
            except Exception:
                pass

        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue = queue.Queue(maxsize=20)
        self.worker: Optional[StationWorker] = None
        self.last_photo = None
        self.video_path = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Select a video file.")
        self.metrics_var = tk.StringVar(value="Video: -- | Faces: 0 | Known: 0 | Unknown: 0 | FPS: 0")
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_text_var = tk.StringVar(value="1.00x")
        self.paused = False
        self.seek_token = 0

        self._build_ui()
        self._poll_queues()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=16, style="App.TFrame")
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        controls = ttk.Frame(root, style="Header.TFrame", padding=(14, 12))
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Video file", style="HeaderSub.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.video_path).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(controls, text="Browse", command=self.browse_video).grid(row=0, column=2, sticky="ew")

        buttons = ttk.Frame(controls, style="Toolbar.TFrame")
        buttons.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.start_btn = ttk.Button(buttons, text="Start video FR", style="Accent.TButton", command=self.start_video)
        self.pause_btn = ttk.Button(buttons, text="Pause", style="Secondary.TButton", command=self.toggle_pause, state=tk.DISABLED)
        self.stop_btn = ttk.Button(buttons, text="Stop", style="Danger.TButton", command=self.stop_video, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(buttons, text="Speed", style="Card.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self.speed_scale = ttk.Scale(
            buttons,
            from_=self.SPEED_MIN,
            to=self.SPEED_MAX,
            variable=self.speed_var,
            command=lambda _v: self.on_speed_changed(),
            length=220,
        )
        self.speed_scale.pack(side=tk.LEFT)
        ttk.Label(buttons, textvariable=self.speed_text_var, style="Card.TLabel", width=7).pack(side=tk.LEFT, padx=(8, 8))
        for value in (0.5, 1.0, 2.0, 4.0):
            ttk.Button(buttons, text=f"{value:g}x", width=4, style="Ghost.TButton", command=lambda v=value: self.set_speed(v)).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Label(buttons, text="Seek", style="Card.TLabel").pack(side=tk.LEFT, padx=(14, 6))
        self.seek_buttons = []
        for label, seconds in (("-60s", -60), ("-10s", -10), ("+10s", 10), ("+60s", 60)):
            btn = ttk.Button(buttons, text=label, width=5, style="Ghost.TButton", command=lambda s=seconds: self.seek_video(s), state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self.seek_buttons.append(btn)

        ttk.Label(
            controls,
            text="Uses the current embeddings, Moodle/GPU options, profile, zoom and recognition controls from the main window.",
            style="MutedCard.TLabel",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(controls, textvariable=self.status_var, style="StatusPill.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        video_card = ttk.Frame(root, style="Card.TFrame", padding=10)
        video_card.grid(row=1, column=0, sticky="nsew")
        video_card.columnconfigure(0, weight=1)
        video_card.rowconfigure(0, weight=1)
        self.video_label = ttk.Label(video_card, text="Start video FR to show processed frames", anchor=tk.CENTER, style="Video.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")
        ttk.Label(video_card, textvariable=self.metrics_var, style="MutedCard.TLabel").grid(row=1, column=0, sticky="ew", pady=(10, 0))

    def browse_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.m4v *.wmv *.webm"), ("All files", "*.*")],
        )
        if path:
            self.video_path.set(path)

    def _current_speed(self) -> float:
        try:
            return max(self.SPEED_MIN, min(self.SPEED_MAX, float(self.speed_var.get())))
        except Exception:
            return 1.0

    def set_speed(self, value: float) -> None:
        self.speed_var.set(max(self.SPEED_MIN, min(self.SPEED_MAX, float(value))))
        self.on_speed_changed()

    def on_speed_changed(self) -> None:
        speed = self._current_speed()
        self.speed_text_var.set(f"{speed:.2f}x")
        if self.worker:
            self.worker.update_controls(playback_speed=speed)

    def _push_main_controls_to_worker(self) -> None:
        if not self.worker:
            return
        parent = self.parent_app
        zoom_factor = float(parent.zoom_factor.get())
        zoom_enabled = bool(parent.zoom_enabled.get()) and zoom_factor > 1.001
        self.worker.update_controls(
            zoom_enabled=zoom_enabled,
            zoom_factor=zoom_factor,
            zoom_center_x=float(parent.zoom_x.get()),
            zoom_center_y=float(parent.zoom_y.get()),
            draw_unknown=bool(parent.draw_unknown.get()),
            grid_search=bool(parent.grid_search.get()),
            hide_confirmed_known=bool(parent.hide_confirmed_known.get()),
            confirmed_similarity_threshold=float(parent.confirmed_similarity.get()),
            confirmed_stable_frames=station.STABLE_FRAMES_REQUIRED,
            hide_confirmed_unknown=bool(parent.hide_confirmed_unknown.get()),
            confirmed_unknown_frames=max(1, int(round(float(parent.confirmed_unknown_frames.get())))),
            skip_resolved_recognition=bool(parent.skip_resolved_recognition.get()),
            max_recognitions_per_frame=max(0, int(round(float(parent.max_recognitions_per_frame.get())))),
            known_stop_after_detections=max(0, int(round(float(parent.known_stop_after_detections.get())))),
            known_database_disabled=bool(getattr(parent, "known_database_disabled", False)),
            playback_paused=self.paused,
            playback_speed=self._current_speed(),
        )

    def start_video(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        path = self.video_path.get().strip()
        if not path:
            messagebox.showwarning("No video selected", "Select a video file first.", parent=self)
            return
        if not Path(path).exists():
            messagebox.showerror("Video not found", f"The selected video file does not exist:\n\n{path}", parent=self)
            return
        if PIL_IMPORT_ERROR is not None:
            messagebox.showerror("Missing Pillow", "Install Pillow first: py -m pip install pillow", parent=self)
            return

        parent = self.parent_app
        selected_profile = parent.profile.get()
        try:
            if getattr(parent, "station_adapter", None) is not None:
                parent.station_adapter.apply_profile(selected_profile, source="desktop-video-window")
            else:
                station.apply_algorithm_profile(selected_profile, source="desktop-video-window")
        except Exception:
            pass

        self.paused = False
        options = RuntimeOptions(
            source_kind="video_file",
            source_value=path,
            embeddings_source=parent.embeddings_source.get(),
            profile=selected_profile,
            use_gpu=bool(parent.use_gpu.get()),
            use_moodle=bool(parent.use_moodle.get()),
            rtsp_transport="tcp",
            playback_speed=self._current_speed(),
            known_database_disabled=bool(getattr(parent, "known_database_disabled", False)),
        )
        self.worker = StationWorker(options, self.frame_queue, self.status_queue, parent.registry)
        self._push_main_controls_to_worker()
        self.worker.start()
        self.start_btn.configure(state=tk.DISABLED)
        self.pause_btn.configure(state=tk.NORMAL, text="Pause")
        self.stop_btn.configure(state=tk.NORMAL)
        for btn in getattr(self, "seek_buttons", []):
            btn.configure(state=tk.NORMAL)
        self.status_var.set("Starting video recognition...")

    def toggle_pause(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        self.paused = not self.paused
        self.pause_btn.configure(text="Resume" if self.paused else "Pause")
        self.worker.update_controls(playback_paused=self.paused, playback_speed=self._current_speed())
        self.status_var.set("Video paused" if self.paused else "Video playing")

    def seek_video(self, delta_seconds: float) -> None:
        if not self.worker or not self.worker.is_alive():
            return
        self.seek_token += 1
        self.worker.update_controls(
            seek_token=self.seek_token,
            seek_delta_seconds=float(delta_seconds),
            seek_absolute_frame=None,
            playback_paused=self.paused,
            playback_speed=self._current_speed(),
        )
        direction = "forward" if delta_seconds > 0 else "back"
        self.status_var.set(f"Seeking {direction} {abs(int(delta_seconds))} seconds...")

    def stop_video(self) -> None:
        if self.worker:
            self.worker.request_stop()
        self.paused = False
        self.start_btn.configure(state=tk.NORMAL)
        self.pause_btn.configure(state=tk.DISABLED, text="Pause")
        self.stop_btn.configure(state=tk.DISABLED)
        for btn in getattr(self, "seek_buttons", []):
            btn.configure(state=tk.DISABLED)

    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        if Image is None or ImageTk is None:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
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

    @staticmethod
    def _format_video_time(position_ms: float) -> str:
        total_seconds = max(0, int(position_ms / 1000.0))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:d}:{seconds:02d}"

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
            self._show_frame(frame)
            pos_frame = int(meta.get("video_position_frame", 0) or 0)
            total_frames = int(meta.get("video_total_frames", 0) or 0)
            pos_text = self._format_video_time(float(meta.get("video_position_ms", 0.0) or 0.0))
            if total_frames > 0:
                progress = f"{pos_frame}/{total_frames} frames"
            else:
                progress = f"frame {pos_frame}"
            self.metrics_var.set(
                f"Video: {pos_text} ({progress}) | Speed: {float(meta.get('playback_speed', self._current_speed())):.2f}x | "
                f"Faces: {meta.get('total_faces', 0)} | Known: {meta.get('known_count', 0)} | "
                f"Unknown: {meta.get('unknown_count', 0)} | FPS: {meta.get('fps', 0.0):.1f}"
            )
            if meta.get("known_report") is not None or meta.get("unknown_report") is not None:
                self.parent_app.refresh_reports(meta)
        except queue.Empty:
            pass

        if self.worker and self.worker.is_alive():
            self._push_main_controls_to_worker()
        elif self.worker:
            self.start_btn.configure(state=tk.NORMAL)
            self.pause_btn.configure(state=tk.DISABLED, text="Pause")
            self.stop_btn.configure(state=tk.DISABLED)
            for btn in getattr(self, "seek_buttons", []):
                btn.configure(state=tk.DISABLED)
            self.paused = False

        self.after(33, self._poll_queues)

    def on_close(self) -> None:
        self.stop_video()
        if getattr(self.parent_app, "video_window", None) is self:
            self.parent_app.video_window = None
        self.destroy()


class DesktopStationApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FaceAttendance Desktop Station")
        self.geometry("1500x900")
        self.minsize(1180, 720)
        palette_bg = getattr(MODERN_PALETTE, "bg", "#101418") if MODERN_PALETTE is not None else "#101418"
        self.configure(bg=palette_bg)

        self.station_adapter = LegacyStationModuleAdapter(station) if LegacyStationModuleAdapter is not None else None
        self.main_playback = VideoPlaybackController() if VideoPlaybackController is not None else None

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
        self.unknown_review_summary_var = tk.StringVar(value="Detected now: 0 persons | Review list: 0 total, 0 labeled, 0 unlabeled | Pictures: 0")
        self.worker: Optional[StationWorker] = None
        self.video_window: Optional["VideoRecognitionWindow"] = None
        self.known_database_disabled = False
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
        if apply_modern_theme is not None:
            apply_modern_theme(self)
            return

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
        root = ttk.Frame(self, padding=16, style="App.TFrame")
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root, style="Header.TFrame", padding=(18, 14))
        header.pack(fill=tk.X, pady=(0, 12))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        title_stack = ttk.Frame(header, style="Header.TFrame")
        title_stack.grid(row=0, column=0, sticky="w")
        ttk.Label(title_stack, text="FaceAttendance Station", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(
            title_stack,
            text="Camera, RTSP and video-file recognition workstation",
            style="HeaderSub.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        # Keep the essential runtime controls outside the long settings panel.
        # They remain visible even when the right-side settings list overflows.
        actions = ttk.Frame(header, style="Toolbar.TFrame")
        actions.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.playback_speed_choice = tk.StringVar(value="1x")
        self.video_paused = tk.BooleanVar(value=False)
        self.video_seek_token = 0
        ttk.Label(actions, text="Video", style="HeaderSub.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.playback_speed_box = ttk.Combobox(
            actions,
            textvariable=self.playback_speed_choice,
            state="readonly",
            values=("0.25x", "0.5x", "1x", "1.5x", "2x", "4x"),
            width=6,
        )
        self.playback_speed_box.pack(side=tk.LEFT, padx=(0, 6))
        self.playback_speed_box.bind("<<ComboboxSelected>>", lambda _event: self.on_playback_speed_changed())
        self.rtsp_transport = tk.StringVar(value="tcp")
        self.video_seek_buttons = []
        for label, seconds in (("-60s", -60), ("-10s", -10), ("+10s", 10), ("+60s", 60)):
            btn = ttk.Button(actions, text=label, width=5, style="Ghost.TButton", command=lambda s=seconds: self.seek_main_video(s), state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=(0, 4))
            self.video_seek_buttons.append(btn)
        self.video_pause_btn = ttk.Button(actions, text="Pause", width=10, style="Secondary.TButton", command=self.toggle_video_pause, state=tk.DISABLED)
        self.video_pause_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.start_btn = ttk.Button(actions, text="Start", style="Accent.TButton", width=12, command=self.start_station)
        self.stop_btn = ttk.Button(actions, text="Stop", style="Danger.TButton", width=12, command=self.stop_station, state=tk.DISABLED)
        self.video_btn = ttk.Button(actions, text="Video window", style="Secondary.TButton", width=14, command=self.open_video_window)
        self.video_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.start_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.stop_btn.pack(side=tk.LEFT)

        status_row = ttk.Frame(root, style="App.TFrame")
        status_row.pack(fill=tk.X, pady=(0, 10))
        status_row.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_row, text="Runtime status", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(status_row, textvariable=self.status_var, style="StatusPill.TLabel").grid(row=0, column=1, sticky="e")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        camera_tab = ttk.Frame(self.notebook, style="App.TFrame")
        reports_tab = ttk.Frame(self.notebook, style="App.TFrame")
        self.notebook.add(camera_tab, text="Camera Station")
        self.notebook.add(reports_tab, text="Reports & Review")

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

        palette = MODERN_PALETTE
        canvas = tk.Canvas(
            outer,
            width=420,
            bg=getattr(palette, "surface", "#161d24"),
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")

        side = ttk.Frame(canvas, style="Card.TFrame", padding=18)
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

        self.video_label = ttk.Label(video_card, text="Start the station to show the camera feed", anchor=tk.CENTER, style="Video.TLabel")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        metrics_bar = ttk.Frame(video_card, style="Card.TFrame")
        metrics_bar.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        for col in range(5):
            metrics_bar.columnconfigure(col, weight=1)
        self.metric_faces_var = tk.StringVar(value="0")
        self.metric_known_var = tk.StringVar(value="0")
        self.metric_unknown_var = tk.StringVar(value="0")
        self.metric_fps_var = tk.StringVar(value="0.0")
        self.metric_video_var = tk.StringVar(value="--")
        metric_factory = metric_card if metric_card is not None else None
        if metric_factory is not None:
            metric_factory(metrics_bar, "Faces", self.metric_faces_var).grid(row=0, column=0, sticky="ew", padx=(0, 6))
            metric_factory(metrics_bar, "Known", self.metric_known_var).grid(row=0, column=1, sticky="ew", padx=6)
            metric_factory(metrics_bar, "Unknown", self.metric_unknown_var).grid(row=0, column=2, sticky="ew", padx=6)
            metric_factory(metrics_bar, "FPS", self.metric_fps_var).grid(row=0, column=3, sticky="ew", padx=6)
            metric_factory(metrics_bar, "Video", self.metric_video_var, width=14).grid(row=0, column=4, sticky="ew", padx=(6, 0))

        self.metrics_var = tk.StringVar(value="Pipeline ready")
        ttk.Label(video_card, textvariable=self.metrics_var, style="MutedCard.TLabel").grid(row=2, column=0, sticky="ew", pady=(8, 0))

        side = self._create_scrollable_settings_panel(parent)
        side.columnconfigure(1, weight=1)

        ttk.Label(side, text="Input source", style="SectionTitle.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        ttk.Label(side, text="Choose camera index, RTSP/HTTP stream or local video file.", style="MutedCard.TLabel", wraplength=360).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self.source_kind = tk.StringVar(value="camera")
        ttk.Radiobutton(side, text="Camera index", variable=self.source_kind, value="camera", command=self.on_source_kind_changed).grid(row=2, column=0, sticky="w")
        ttk.Radiobutton(side, text="RTSP/HTTP stream", variable=self.source_kind, value="stream", command=self.on_source_kind_changed).grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(side, text="Video file", variable=self.source_kind, value="video_file", command=self.on_source_kind_changed).grid(row=3, column=1, columnspan=2, sticky="w")

        ttk.Label(side, text="Source", style="Card.TLabel").grid(row=4, column=0, sticky="w", pady=(12, 0))
        ttk.Label(side, text="RTSP transport", style="Card.TLabel").grid(row=4, column=1, sticky="e", padx=(8, 6), pady=(12, 0))
        self.rtsp_transport_box = ttk.Combobox(
            side,
            textvariable=self.rtsp_transport,
            state="readonly",
            values=("tcp", "udp"),
            width=7,
        )
        self.rtsp_transport_box.grid(row=4, column=2, sticky="ew", pady=(12, 0))
        self.source_value = tk.StringVar(value="0")
        ttk.Entry(side, textvariable=self.source_value, width=36).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(side, text="Browse file", command=self.browse_source_file).grid(row=5, column=2, sticky="ew", padx=(6, 0), pady=(4, 0))
        self.source_hint_var = tk.StringVar(value="Camera index example: 0 or 1")
        ttk.Label(side, textvariable=self.source_hint_var, style="Muted.TLabel", wraplength=340).grid(row=6, column=0, columnspan=3, sticky="w", pady=(6, 0))

        ttk.Label(side, text="Embeddings source", style="Card.TLabel").grid(row=7, column=0, sticky="w", pady=(12, 0))
        self.embeddings_source = tk.StringVar(value=str(APP_DIR / "images"))
        ttk.Entry(side, textvariable=self.embeddings_source, width=36).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(side, text="Browse", command=self.browse_embeddings).grid(row=8, column=2, sticky="ew", padx=(6, 0), pady=(4, 0))

        ttk.Label(side, text="Profile", style="Card.TLabel").grid(row=9, column=0, sticky="w", pady=(12, 0))
        self.profile = tk.StringVar(value="walkthrough_realtime" if "walkthrough_realtime" in station.ALGORITHM_PROFILES else ("crowd_extreme" if "crowd_extreme" in station.ALGORITHM_PROFILES else ("crowd_turbo" if "crowd_turbo" in station.ALGORITHM_PROFILES else ("crowd_fast" if "crowd_fast" in station.ALGORITHM_PROFILES else "fast_short"))))
        profile_box = ttk.Combobox(side, textvariable=self.profile, state="readonly", values=list(station.ALGORITHM_PROFILES.keys()), width=34)
        profile_box.grid(row=10, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        self.use_gpu = tk.BooleanVar(value=True)
        self.use_moodle = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Try GPU acceleration (ONNX Runtime CUDA for SFace)", variable=self.use_gpu).grid(row=11, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Checkbutton(side, text="Load roster/embeddings from Moodle API", variable=self.use_moodle).grid(row=12, column=0, columnspan=3, sticky="w", pady=(4, 0))

        ttk.Separator(side).grid(row=13, column=0, columnspan=3, sticky="ew", pady=14)
        ttk.Label(side, text="Digital zoom", style="SectionTitle.TLabel").grid(row=14, column=0, columnspan=3, sticky="w")

        self.zoom_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="Enable zoom before recognition", variable=self.zoom_enabled, command=self.toggle_zoom).grid(row=15, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.zoom_factor = tk.DoubleVar(value=1.0)
        ttk.Label(side, text="Zoom factor", style="Card.TLabel").grid(row=16, column=0, sticky="w", pady=(10, 0))
        ttk.Scale(side, from_=1.0, to=8.0, variable=self.zoom_factor, command=lambda _v: self.on_zoom_factor_changed()).grid(row=17, column=0, columnspan=3, sticky="ew")
        self.zoom_status_var = tk.StringVar(value="Zoom OFF")
        ttk.Label(side, textvariable=self.zoom_status_var, style="Card.TLabel").grid(row=16, column=1, columnspan=2, sticky="e", pady=(10, 0))

        self.zoom_x = tk.DoubleVar(value=0.5)
        self.zoom_y = tk.DoubleVar(value=0.5)
        ttk.Label(side, text="Horizontal center", style="Card.TLabel").grid(row=18, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0.0, to=1.0, variable=self.zoom_x, command=lambda _v: self.push_controls()).grid(row=19, column=0, columnspan=3, sticky="ew")
        ttk.Label(side, text="Vertical center", style="Card.TLabel").grid(row=20, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0.0, to=1.0, variable=self.zoom_y, command=lambda _v: self.push_controls()).grid(row=21, column=0, columnspan=3, sticky="ew")

        pan = ttk.Frame(side, style="Card.TFrame")
        pan.grid(row=22, column=0, columnspan=3, sticky="ew", pady=(8, 0))
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
        ttk.Checkbutton(side, text="Draw unknown faces", variable=self.draw_unknown, command=self.push_controls).grid(row=23, column=0, columnspan=3, sticky="w", pady=(12, 0))
        ttk.Checkbutton(side, text="Periodic grid search / slower high recall", variable=self.grid_search, command=self.push_controls).grid(row=24, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Checkbutton(side, text="Hide confirmed known faces", variable=self.hide_confirmed_known, command=self.push_controls).grid(row=25, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Label(side, text="Confirmed known similarity", style="Card.TLabel").grid(row=26, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0.36, to=0.80, variable=self.confirmed_similarity, command=lambda _v: self.push_controls()).grid(row=27, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(side, text="Stop repeated unknown faces", variable=self.hide_confirmed_unknown, command=self.push_controls).grid(row=28, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(side, text="Stop Unknown after detections", style="Card.TLabel").grid(row=29, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=3, to=30, variable=self.confirmed_unknown_frames, command=lambda _v: self.push_controls()).grid(row=30, column=0, columnspan=3, sticky="ew")
        ttk.Checkbutton(side, text="Skip recognition for resolved faces", variable=self.skip_resolved_recognition, command=self.push_controls).grid(row=31, column=0, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Label(side, text="Max SFace recognitions/frame (0 = all)", style="Card.TLabel").grid(row=32, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0, to=40, variable=self.max_recognitions_per_frame, command=lambda _v: self.push_controls()).grid(row=33, column=0, columnspan=3, sticky="ew")
        ttk.Label(side, text="Stop known after detections", style="Card.TLabel").grid(row=34, column=0, sticky="w", pady=(8, 0))
        ttk.Scale(side, from_=0, to=300, variable=self.known_stop_after_detections, command=lambda _v: self.push_controls()).grid(row=35, column=0, columnspan=3, sticky="ew")
        ttk.Label(side, text="0 = disabled. 150 is recommended for classroom tests.", style="Muted.TLabel").grid(row=36, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Label(
            side,
            text="Start/Stop controls are fixed in the top bar. For video files, use Pause, speed and -60s/-10s/+10s/+60s seek controls in the top bar.",
            style="Card.TLabel",
            wraplength=340,
        ).grid(row=37, column=0, columnspan=3, sticky="w", pady=(18, 0))

        self.on_source_kind_changed()

    def _build_reports_tab(self, parent: ttk.Frame) -> None:
        """Build the reports page with Known on the left and Unknown review on the right."""
        parent.columnconfigure(0, weight=3, minsize=560)
        parent.columnconfigure(1, weight=2, minsize=430)
        parent.rowconfigure(1, weight=1)

        top_bar = ttk.Frame(parent, style="Header.TFrame", padding=(14, 12))
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        top_bar.columnconfigure(0, weight=1)
        ttk.Label(top_bar, text="Reports & Unknown Review", style="HeaderTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(top_bar, text="Refresh report", style="Secondary.TButton", command=self.refresh_reports).grid(row=0, column=1, sticky="e", padx=(8, 0))
        ttk.Button(top_bar, text="Export CSV", style="Accent.TButton", command=self.export_report_csv).grid(row=0, column=2, sticky="e", padx=(8, 0))

        known_section = ttk.Frame(parent)
        known_section.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        known_section.rowconfigure(1, weight=1)
        known_section.columnconfigure(0, weight=1)
        known_header = ttk.Frame(known_section)
        known_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        known_header.columnconfigure(0, weight=1)
        ttk.Label(known_header, text="Known persons", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(known_header, text="Delete ALL known faces", style="Danger.TButton", command=self.delete_all_known_faces).grid(row=0, column=1, sticky="e")

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
        if configure_tree_tags is not None:
            configure_tree_tags(self.known_tree)

        unknown_section = ttk.Frame(parent, style="Card.TFrame")
        unknown_section.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        unknown_section.rowconfigure(2, weight=1)
        unknown_section.columnconfigure(0, weight=1)
        unknown_title_bar = ttk.Frame(unknown_section)
        unknown_title_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        unknown_title_bar.columnconfigure(1, weight=1)
        ttk.Label(unknown_title_bar, text="Unknown review", style="SectionTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            unknown_title_bar,
            textvariable=self.unknown_review_summary_var,
            style="MutedCard.TLabel",
            anchor=tk.E,
        ).grid(row=0, column=1, sticky="ew", padx=(12, 0))

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

        ttk.Button(unknown_tools, text="Delete selected unknowns", style="Danger.TButton", command=self.delete_selected_unknown).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(7, 0), padx=(0, 4))
        ttk.Button(unknown_tools, text="Delete all unlabeled", style="Danger.TButton", command=self.delete_all_unlabeled_unknown).grid(row=1, column=2, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(unknown_tools, text="Export unknown + labeled faces", style="Accent.TButton", command=self.export_unknown_faces_package).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(7, 0), padx=(0, 4))
        ttk.Button(unknown_tools, text="Import unknown faces", style="Secondary.TButton", command=self.import_unknown_faces_package).grid(row=2, column=2, columnspan=2, sticky="ew", pady=(7, 0))
        ttk.Button(unknown_tools, text="Delete records without pictures", style="Secondary.TButton", command=self.delete_unknowns_without_pictures).grid(row=3, column=0, columnspan=4, sticky="ew", pady=(7, 0))

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
        if configure_tree_tags is not None:
            configure_tree_tags(self.unknown_tree)
        self._update_unknown_sort_headings()

        review_bar = ttk.Frame(unknown_card, style="Card.TFrame")
        review_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        review_bar.columnconfigure(1, weight=1)
        ttk.Label(review_bar, text="Assign selected unknown as", style="Card.TLabel").grid(row=0, column=0, sticky="w")
        self.assign_label_var = tk.StringVar()
        ttk.Entry(review_bar, textvariable=self.assign_label_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(review_bar, text="Assign label", style="Accent.TButton", command=self.assign_unknown_label).grid(row=0, column=2, sticky="ew")

        preview_frame = ttk.Frame(unknown_card, style="Card.TFrame")
        preview_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.preview_label = ttk.Label(
            preview_frame,
            text="Select an unknown track to preview the last captured picture.",
            style="Card.TLabel",
            anchor=tk.CENTER,
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def open_video_window(self) -> None:
        if self.video_window is not None and self.video_window.winfo_exists():
            self.video_window.lift()
            self.video_window.focus_force()
            return
        self.video_window = VideoRecognitionWindow(self)

    def browse_source_file(self) -> None:
        path = filedialog.askopenfilename(title="Select video file", filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov *.m4v *.webm"), ("All files", "*.*")])
        if path:
            self.source_kind.set("video_file")
            self.source_value.set(path)
            self.on_source_kind_changed()

    def browse_embeddings(self) -> None:
        path = filedialog.askdirectory(title="Select embeddings folder")
        if path:
            self.embeddings_source.set(path)
            self.known_database_disabled = False
            self.push_controls()

    def _selected_playback_speed(self) -> float:
        raw = str(getattr(self, "playback_speed_choice", tk.StringVar(value="1x")).get() or "1x")
        if parse_speed_choice is not None:
            return parse_speed_choice(raw, default=1.0)
        raw = raw.strip().lower()
        raw = raw[:-1] if raw.endswith("x") else raw
        try:
            return max(0.10, min(8.0, float(raw)))
        except ValueError:
            return 1.0

    def _source_value_is_local_file(self) -> bool:
        value = str(getattr(self, "source_value", tk.StringVar(value="")).get() or "").strip()
        if not value or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value):
            return False
        try:
            return Path(value).exists()
        except Exception:
            return False

    def _selected_source_is_video_file(self) -> bool:
        kind = str(getattr(self, "source_kind", tk.StringVar(value="camera")).get() or "camera")
        return kind == "video_file" or (kind == "stream" and self._source_value_is_local_file())

    def on_source_kind_changed(self) -> None:
        if not hasattr(self, "source_hint_var"):
            return
        kind = self.source_kind.get()
        if kind == "camera":
            self.source_hint_var.set("Camera index example: 0 or 1")
        elif kind == "stream":
            self.source_hint_var.set("RTSP/HTTP stream example: rtsp://user:pass@ip:554/Streaming/Channels/101. Use TCP for most NVR/IP cameras; try UDP only if TCP is slow.")
        else:
            self.source_hint_var.set("Video file source: browse or paste a local .mp4/.avi/.mkv/.mov path. Recognition uses the same FR pipeline as the live camera.")
        if hasattr(self, "rtsp_transport_box"):
            self.rtsp_transport_box.configure(state="readonly" if kind == "stream" else tk.DISABLED)
        self._refresh_video_playback_controls()

    def _refresh_video_playback_controls(self) -> None:
        if not hasattr(self, "video_pause_btn"):
            return
        is_running = bool(self.worker and self.worker.is_alive())
        is_video = self._selected_source_is_video_file()
        if is_running and is_video:
            self.video_pause_btn.configure(state=tk.NORMAL)
            seek_state = tk.NORMAL
        else:
            self.video_pause_btn.configure(state=tk.DISABLED)
            seek_state = tk.DISABLED
            self.video_paused.set(False)
        for btn in getattr(self, "video_seek_buttons", []):
            btn.configure(state=seek_state)
        self.video_pause_btn.configure(text="Resume video" if bool(self.video_paused.get()) else "Pause video")

    def on_playback_speed_changed(self) -> None:
        selected_speed = self._selected_playback_speed()
        if self.main_playback is not None:
            self.main_playback.set_speed(selected_speed)
        if self.worker:
            self.worker.update_controls(playback_speed=selected_speed)
        self._refresh_video_playback_controls()

    def toggle_video_pause(self) -> None:
        if not (self.worker and self.worker.is_alive() and self._selected_source_is_video_file()):
            return
        if self.main_playback is not None:
            self.main_playback.paused = bool(self.video_paused.get())
            paused = self.main_playback.toggle_pause()
        else:
            paused = not bool(self.video_paused.get())
        self.video_paused.set(paused)
        self.worker.update_controls(playback_paused=paused, playback_speed=self._selected_playback_speed())
        self.video_pause_btn.configure(text="Resume video" if paused else "Pause video")
        self.status_var.set("Video paused" if paused else "Video resumed")

    def seek_main_video(self, delta_seconds: float) -> None:
        if not (self.worker and self.worker.is_alive() and self._selected_source_is_video_file()):
            return
        if self.main_playback is not None:
            self.video_seek_token = self.main_playback.seek()
        else:
            self.video_seek_token += 1
        self.worker.update_controls(
            seek_token=self.video_seek_token,
            seek_delta_seconds=float(delta_seconds),
            seek_absolute_frame=None,
            playback_paused=bool(self.video_paused.get()),
            playback_speed=self._selected_playback_speed(),
        )
        direction = "forward" if delta_seconds > 0 else "back"
        self.status_var.set(f"Seeking video {direction} {abs(int(delta_seconds))} seconds...")

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
            if self.station_adapter is not None:
                self.station_adapter.apply_profile(selected_profile, source="desktop-ui-controls")
            else:
                station.apply_algorithm_profile(selected_profile, source="desktop-ui-controls")
        except Exception:
            pass

        self.video_paused.set(False)
        if self.main_playback is not None:
            self.main_playback.stop()
            self.main_playback.set_speed(self._selected_playback_speed())
        self._refresh_video_playback_controls()

        options = RuntimeOptions(
            source_kind=self.source_kind.get(),
            source_value=self.source_value.get(),
            embeddings_source=self.embeddings_source.get(),
            profile=selected_profile,
            use_gpu=bool(self.use_gpu.get()),
            use_moodle=bool(self.use_moodle.get()),
            rtsp_transport=str(getattr(self, "rtsp_transport", tk.StringVar(value="tcp")).get() or "tcp"),
            playback_speed=self._selected_playback_speed(),
            known_database_disabled=bool(getattr(self, "known_database_disabled", False)),
        )
        self.worker = StationWorker(options, self.frame_queue, self.status_queue, self.registry)
        self.push_controls()
        self.worker.start()
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self._refresh_video_playback_controls()
        self.status_var.set("Starting...")

    def stop_station(self) -> None:
        if self.worker:
            self.worker.request_stop()
        self.video_paused.set(False)
        if self.main_playback is not None:
            self.main_playback.stop()
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self._refresh_video_playback_controls()

    def _effective_zoom_enabled(self) -> bool:
        return bool(self.zoom_enabled.get()) and float(self.zoom_factor.get()) > 1.001

    def _refresh_zoom_status_label(self) -> None:
        if hasattr(self, "zoom_status_var"):
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
        # Moving the zoom slider above 1x should automatically enable zoom.
        # Moving it back to 1x should turn zoom off.
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
                hide_confirmed_known=bool(self.hide_confirmed_known.get()),
                confirmed_similarity_threshold=float(self.confirmed_similarity.get()),
                confirmed_stable_frames=station.STABLE_FRAMES_REQUIRED,
                hide_confirmed_unknown=bool(self.hide_confirmed_unknown.get()),
                confirmed_unknown_frames=max(1, int(round(float(self.confirmed_unknown_frames.get())))),
                skip_resolved_recognition=bool(self.skip_resolved_recognition.get()),
                max_recognitions_per_frame=max(0, int(round(float(self.max_recognitions_per_frame.get())))),
                known_stop_after_detections=max(0, int(round(float(self.known_stop_after_detections.get())))),
                known_database_disabled=bool(getattr(self, "known_database_disabled", False)),
                playback_paused=bool(self.video_paused.get()),
                playback_speed=self._selected_playback_speed(),
            )

    def pan_zoom(self, dx: float, dy: float) -> None:
        if float(self.zoom_factor.get()) <= 1.001:
            self.zoom_factor.set(2.0)
        self.zoom_enabled.set(True)
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
            metrics_text = (
                f"Faces: {meta.get('total_faces', 0)} | Remaining: {meta.get('remaining_faces', meta.get('total_faces', 0))} | "
                f"Known: {meta.get('known_count', 0)} | Unknown: {meta.get('unknown_count', 0)} | "
                f"Hidden: {meta.get('hidden_confirmed_count', 0)} | "
                f"SFace: {meta.get('sface_calls', 0)}/{meta.get('sface_batches', 0)} | FPS: {meta.get('fps', 0.0):.1f} | "
            )
            if meta.get("is_video_playback"):
                pos_text = VideoRecognitionWindow._format_video_time(float(meta.get("video_position_ms", 0.0) or 0.0))
                pos_frame = int(meta.get("video_position_frame", 0) or 0)
                total_frames = int(meta.get("video_total_frames", 0) or 0)
                speed = float(meta.get("playback_speed", self._selected_playback_speed()) or 1.0)
                paused_text = " PAUSED" if bool(meta.get("playback_paused", False)) else ""
                if total_frames > 0:
                    metrics_text += f"Video: {pos_text} frame {pos_frame}/{total_frames} | Speed: {speed:.2f}x{paused_text} | "
                else:
                    metrics_text += f"Video: {pos_text} | Speed: {speed:.2f}x{paused_text} | "
            metrics_text += (
                f"ms read {meta.get('read_ms', 0.0):.1f} det {meta.get('detect_ms', 0.0):.1f} "
                f"sface {meta.get('sface_ms', 0.0):.1f} align {meta.get('sface_align_ms', 0.0):.1f} "
                f"infer {meta.get('sface_infer_ms', 0.0):.1f} track {meta.get('track_ms', 0.0):.1f} "
                f"draw {meta.get('draw_ms', 0.0):.1f} loop {meta.get('loop_ms', 0.0):.1f}"
            )
            self.metrics_var.set(metrics_text)
            if hasattr(self, "metric_faces_var"):
                self.metric_faces_var.set(str(meta.get("total_faces", 0)))
                self.metric_known_var.set(str(meta.get("known_count", 0)))
                self.metric_unknown_var.set(str(meta.get("unknown_count", 0)))
                self.metric_fps_var.set(f"{float(meta.get('fps', 0.0) or 0.0):.1f}")
                if meta.get("is_video_playback"):
                    self.metric_video_var.set(f"{VideoRecognitionWindow._format_video_time(float(meta.get('video_position_ms', 0.0) or 0.0))} @ {float(meta.get('playback_speed', 1.0) or 1.0):.2f}x")
                else:
                    self.metric_video_var.set("live")
            if meta.get("known_report") is not None or meta.get("unknown_report") is not None:
                self.refresh_reports(meta)
        except queue.Empty:
            pass

        if self.worker and not self.worker.is_alive():
            self.start_btn.configure(state=tk.NORMAL)
            self.stop_btn.configure(state=tk.DISABLED)
            self._refresh_video_playback_controls()

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

    def _update_unknown_review_summary(self, meta: Optional[dict], unknown_rows: List[dict]) -> None:
        if not hasattr(self, "unknown_review_summary_var"):
            return
        meta = meta or {}
        rows = list(unknown_rows or [])
        review_total = len(rows)
        labeled = sum(1 for row in rows if str(row.get("assigned_label") or "").strip())
        unlabeled = max(0, review_total - labeled)
        pictures = sum(len(row.get("captures", []) or []) for row in rows)
        detected_now = int(meta.get("total_faces", 0) or 0)
        known_now = int(meta.get("known_count", 0) or 0)
        unknown_now = int(meta.get("unknown_count", 0) or 0)
        self.unknown_review_summary_var.set(
            f"Detected now: {detected_now} persons "
            f"(known {known_now}, unknown {unknown_now}) | "
            f"Review list: {review_total} total, {labeled} labeled, {unlabeled} unlabeled | "
            f"Pictures: {pictures}"
        )

    def refresh_reports(self, meta: Optional[dict] = None) -> None:
        meta = meta or self.latest_meta or {}
        known_rows = meta.get("known_report", [])
        unknown_rows = self.registry.snapshot(sort_by=self.unknown_sort_mode) if not meta.get("unknown_report") else list(meta.get("unknown_report", []))
        unknown_rows = self._sort_unknown_rows(unknown_rows)
        self._update_unknown_sort_headings()
        self._update_unknown_review_summary(meta, unknown_rows)

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
        if retag_tree_rows is not None:
            retag_tree_rows(self.known_tree)

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
        if retag_tree_rows is not None:
            retag_tree_rows(self.unknown_tree)

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

    def _active_workers(self) -> List[StationWorker]:
        workers = []
        if self.worker and self.worker.is_alive():
            workers.append(self.worker)
        video_window = getattr(self, "video_window", None)
        video_worker = getattr(video_window, "worker", None) if video_window is not None else None
        if video_worker and video_worker.is_alive() and video_worker not in workers:
            workers.append(video_worker)
        return workers

    def _register_label_descriptors_in_running_workers(self, label: str, descriptors: List[np.ndarray]) -> int:
        self.known_database_disabled = False
        count = 0
        for worker in self._active_workers():
            worker.register_known(label, descriptors)
            count += 1
        return count

    def _register_reviewed_embeddings_in_running_workers(self) -> int:
        reviewed_features = load_reviewed_known_features()
        if reviewed_features:
            self.known_database_disabled = False
        if not reviewed_features:
            return 0
        added_total = 0
        for label, descriptors in reviewed_features.items():
            for worker in self._active_workers():
                worker.register_known(label, descriptors)
                added_total += len(descriptors or [])
        return added_total

    def _known_embedding_files_from_source(self, source_value: str) -> List[Path]:
        """Collect embedding JSON files from the currently selected local source.

        The project root itself is intentionally refused because it may contain
        station_config.json, unknown_registry.json, and other non-enrollment JSON.
        """
        source_text = str(source_value or "").strip()
        if not source_text:
            return []
        source = Path(source_text).expanduser()
        if not source.exists():
            return []

        try:
            source_resolved = source.resolve()
            app_resolved = APP_DIR.resolve()
        except Exception:
            source_resolved = source
            app_resolved = APP_DIR

        if source_resolved == app_resolved:
            return []

        try:
            candidates = station.find_embedding_files(source) if hasattr(station, "find_embedding_files") else ([source] if source.is_file() else list(source.rglob("*.json")))
        except Exception:
            candidates = [source] if source.is_file() else list(source.rglob("*.json"))

        files: List[Path] = []
        seen = set()
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except Exception:
                resolved = path
            # Never delete Unknown-review registry files through the known-face cleanup.
            if is_path_inside(resolved, UNKNOWN_DIR):
                continue
            key = str(resolved).lower()
            if key in seen:
                continue
            if path.suffix.lower() == ".json" or "embedding" in path.name.lower():
                files.append(path)
                seen.add(key)
        return files

    def _reviewed_embedding_files(self) -> List[Path]:
        if not REVIEWED_EMBEDDINGS_DIR.exists():
            return []
        return [path for path in sorted(REVIEWED_EMBEDDINGS_DIR.rglob("*.json")) if path.is_file()]

    def _project_known_embedding_pool_files(self) -> List[Path]:
        """Collect known embeddings from local project pools, not only the selected source."""
        files: List[Path] = []
        if not APP_DIR.exists():
            return files
        for child in sorted(APP_DIR.iterdir()):
            if not child.is_dir():
                continue
            name = child.name.lower()
            if child.resolve() in {UNKNOWN_DIR.resolve(), REVIEWED_EMBEDDINGS_DIR.resolve(), DELETED_KNOWN_BACKUP_DIR.resolve()}:
                continue
            if "embedding" in name or name in {"images", "known", "known_faces", "known_embeddings"}:
                files.extend(self._known_embedding_files_from_source(str(child)))
        return files

    def _backup_known_embedding_files(self, files: List[Path]) -> Optional[Path]:
        files = [Path(path) for path in files if Path(path).is_file()]
        if not files:
            return None
        DELETED_KNOWN_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        backup_path = DELETED_KNOWN_BACKUP_DIR / f"known_faces_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        used_names = set()
        with zipfile.ZipFile(backup_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "created_at": now_iso(),
                "type": "faceattendance_deleted_known_faces_backup",
                "description": "Automatic backup created before deleting all local known face embeddings.",
                "files": [str(path) for path in files],
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for index, path in enumerate(files, start=1):
                try:
                    resolved = path.resolve()
                except Exception:
                    resolved = path
                try:
                    if is_path_inside(resolved, APP_DIR):
                        arc_name = resolved.relative_to(APP_DIR.resolve()).as_posix()
                    else:
                        arc_name = f"external_embedding_source/{index:04d}_{path.name}"
                except Exception:
                    arc_name = f"external_embedding_source/{index:04d}_{path.name}"
                original_arc_name = arc_name
                suffix_index = 1
                while arc_name in used_names:
                    stem = Path(original_arc_name).stem
                    suffix = Path(original_arc_name).suffix
                    parent = Path(original_arc_name).parent.as_posix()
                    arc_name = f"{parent}/{stem}_{suffix_index}{suffix}" if parent != "." else f"{stem}_{suffix_index}{suffix}"
                    suffix_index += 1
                used_names.add(arc_name)
                zf.write(path, arc_name)
        return backup_path

    def _clear_known_reports_in_ui(self) -> None:
        if self.latest_meta is None:
            self.latest_meta = {}
        self.latest_meta["known_report"] = []
        self.latest_meta["known_count"] = 0
        if hasattr(self, "known_tree"):
            for iid in list(self.known_tree.get_children()):
                self.known_tree.delete(iid)

    def delete_all_known_faces(self) -> None:
        source_files = self._known_embedding_files_from_source(self.embeddings_source.get())
        project_pool_files = self._project_known_embedding_pool_files()
        reviewed_files = self._reviewed_embedding_files()

        files_by_key: Dict[str, Path] = {}
        for path in source_files + project_pool_files + reviewed_files:
            try:
                key = str(path.resolve()).lower()
            except Exception:
                key = str(path).lower()
            files_by_key[key] = path
        files_to_delete = list(files_by_key.values())

        labeled_tracks = sum(1 for row in self.registry.snapshot() if str(row.get("assigned_label") or "").strip())
        active_workers = self._active_workers()

        if not files_to_delete and not active_workers and labeled_tracks == 0:
            messagebox.showinfo("Delete known faces", "There are no local known-face embeddings or active known-face workers to clear.")
            return

        source_note = str(Path(self.embeddings_source.get()).expanduser()) if str(self.embeddings_source.get() or "").strip() else "<empty>"
        moodle_note = (
            "\n\nMoodle mode is enabled: this cannot delete embeddings stored on the Moodle server. "
            "It clears the active worker and local reviewed embeddings; server-side known faces can return after restart unless removed from Moodle."
            if bool(self.use_moodle.get()) else ""
        )
        confirm_text = (
            "Delete ALL known faces from this station?\n\n"
            f"Local embedding source:\n{source_note}\n\n"
            "Also scanned local project folders whose names contain 'embedding'.\n"
            f"Embedding JSON files that will be deleted: {len(files_to_delete)}\n"
            f"Unknown-review labels that will be reset: {labeled_tracks}\n"
            f"Running recognition workers that will be cleared: {len(active_workers)}\n\n"
            "A backup ZIP will be created before deletion. Unknown face pictures are kept for review."
            f"{moodle_note}"
        )
        if not messagebox.askyesno("Delete ALL known faces", confirm_text):
            return

        backup_path = None
        deleted_files = 0
        failed_files = []
        try:
            backup_path = self._backup_known_embedding_files(files_to_delete)
            for path in files_to_delete:
                try:
                    path.unlink()
                    deleted_files += 1
                except Exception as exc:
                    failed_files.append(f"{path}: {exc}")

            reset_labels = self.registry.clear_assigned_labels()
            self.known_database_disabled = True
            workers_cleared = 0
            for worker in active_workers:
                worker.clear_known_database(disable_reload=True)
                workers_cleared += 1
            self._clear_known_reports_in_ui()
            self._force_unknown_report_refresh()
            self.push_controls()

            msg = (
                f"Deleted known embedding files: {deleted_files}\n"
                f"Reset labeled Unknown-review tracks: {reset_labels}\n"
                f"Cleared running workers: {workers_cleared}\n"
            )
            if backup_path:
                msg += f"\nBackup ZIP:\n{backup_path}\n"
            if failed_files:
                sample = "\n".join(failed_files[:8])
                if len(failed_files) > 8:
                    sample += f"\n... +{len(failed_files) - 8} more"
                messagebox.showwarning("Known faces partially deleted", msg + "\nSome files could not be deleted:\n" + sample)
            else:
                messagebox.showinfo("Known faces deleted", msg)
        except Exception as exc:
            messagebox.showerror("Delete known faces failed", str(exc))

    def export_unknown_faces_package(self) -> None:
        default_name = f"review_faces_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        path = filedialog.asksaveasfilename(
            title="Export unknown and labeled review faces",
            defaultextension=".zip",
            initialfile=default_name,
            filetypes=[("ZIP package", "*.zip"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            tracks, images, reviewed, labeled_tracks, labeled_images = self.registry.export_package(Path(path), include_reviewed_embeddings=True)
            unlabeled_tracks = max(0, tracks - labeled_tracks)
            messagebox.showinfo(
                "Review faces exported",
                f"Saved package:\n{path}\n\n"
                f"Review tracks total: {tracks}\n"
                f"Unlabeled unknown tracks: {unlabeled_tracks}\n"
                f"Already labeled tracks: {labeled_tracks}\n"
                f"All review face images: {images}\n"
                f"Labeled face image copies: {labeled_images}\n"
                f"Reviewed/labeled embedding files: {reviewed}\n\n"
                f"Labeled images are also grouped under labeled_faces/<label>/ inside the ZIP.",
            )
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))

    def import_unknown_faces_package(self) -> None:
        path = filedialog.askopenfilename(
            title="Import unknown faces",
            filetypes=[("ZIP package", "*.zip"), ("All files", "*.*")],
        )
        if not path:
            return
        if not messagebox.askyesno(
            "Import unknown faces",
            "Import this package and merge it with the current Unknown review list?\n\n"
            "Existing records are kept. Imported records with duplicate IDs receive a new UNK id.",
        ):
            return
        try:
            tracks, images, reviewed = self.registry.import_package(Path(path))
            self.registry.load()
            added_to_running = self._register_reviewed_embeddings_in_running_workers()
            self._force_unknown_report_refresh()
            messagebox.showinfo(
                "Unknown faces imported",
                f"Imported from:\n{path}\n\n"
                f"Unknown tracks: {tracks}\n"
                f"Face images: {images}\n"
                f"Reviewed/labeled embedding files: {reviewed}\n"
                f"Embeddings added to running workers: {added_to_running}",
            )
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))

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
            updated_workers = self._register_label_descriptors_in_running_workers(label, descriptors)
            self._force_unknown_report_refresh()
            messagebox.showinfo(
                "Label assigned",
                f"{track_id} was assigned to {label}.\n\n"
                f"The embedding was saved permanently under:\n{REVIEWED_EMBEDDINGS_DIR}\n\n"
                f"Running recognition workers updated: {updated_workers}",
            )
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
