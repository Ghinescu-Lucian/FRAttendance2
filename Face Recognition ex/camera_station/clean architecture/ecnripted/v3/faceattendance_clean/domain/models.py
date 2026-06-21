"""Domain-level models for the FaceAttendance station.

This module intentionally has no dependency on Tkinter, OpenCV, Moodle, or the
file system.  It describes the concepts used by the station so that UI code,
worker orchestration and storage/adapters can evolve independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SourceKind(str, Enum):
    CAMERA = "camera"
    STREAM = "stream"
    VIDEO_FILE = "video_file"


@dataclass(frozen=True)
class MediaSource:
    kind: SourceKind
    value: str
    rtsp_transport: str = "tcp"

    @property
    def is_video_file(self) -> bool:
        return self.kind == SourceKind.VIDEO_FILE

    @property
    def is_stream(self) -> bool:
        return self.kind == SourceKind.STREAM


@dataclass(frozen=True)
class PlaybackState:
    paused: bool = False
    speed: float = 1.0
    position_ms: float = 0.0
    position_frame: int = 0
    total_frames: int = 0


@dataclass(frozen=True)
class RecognitionCounters:
    total_faces: int = 0
    known_faces: int = 0
    unknown_faces: int = 0
    hidden_faces: int = 0
    fps: float = 0.0


@dataclass(frozen=True)
class KnownPersonReport:
    name: str
    detections: int = 0
    last_seen: Optional[str] = None
    best_similarity: Optional[float] = None
    status: str = "active"


@dataclass(frozen=True)
class UnknownFaceReport:
    track_id: str
    detections: int = 0
    captures: int = 0
    last_seen: Optional[str] = None
    assigned_label: str = ""
    image_paths: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class StationSnapshot:
    counters: RecognitionCounters = field(default_factory=RecognitionCounters)
    playback: PlaybackState = field(default_factory=PlaybackState)
    known: Dict[str, KnownPersonReport] = field(default_factory=dict)
    unknown: List[UnknownFaceReport] = field(default_factory=list)
