"""Small application use-cases shared by the desktop screens."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoPlaybackController:
    """UI-independent video playback state.

    The GUI can bind buttons and sliders to this class, while the worker receives
    only primitive controls such as paused/speed/seek_token.  This is the first
    extraction point for moving video logic out of Tkinter.
    """

    paused: bool = False
    speed: float = 1.0
    seek_token: int = 0

    def set_speed(self, value: float) -> float:
        self.speed = max(0.10, min(8.0, float(value or 1.0)))
        return self.speed

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def stop(self) -> None:
        self.paused = False

    def seek(self) -> int:
        self.seek_token += 1
        return self.seek_token


def parse_speed_choice(raw: Optional[str], default: float = 1.0) -> float:
    value = str(raw or f"{default}x").strip().lower()
    if value.endswith("x"):
        value = value[:-1]
    try:
        return max(0.10, min(8.0, float(value)))
    except Exception:
        return default
