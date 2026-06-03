#!/usr/bin/env python3
"""
Standalone RTSP viewer for IP cameras.

Usage:
    py rtsp_viewer.py "rtsp://admin:PASSWORD@192.168.0.64:554/Streaming/channels/102"

Controls:
    q / ESC  - quit
    p        - save current frame as JPG
    + / =    - digital zoom in
    - / _    - digital zoom out
    arrows   - pan while zoomed
    r        - reset zoom/pan
"""

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def configure_ffmpeg_transport(transport: str) -> None:
    """
    OpenCV reads OPENCV_FFMPEG_CAPTURE_OPTIONS when opening a VideoCapture.
    TCP is usually much more stable than UDP for RTSP on unreliable LAN/Wi-Fi.
    """
    if transport.lower() == "tcp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|"
            "stimeout;5000000|"
            "max_delay;500000|"
            "fflags;nobuffer|"
            "flags;low_delay|"
            "analyzeduration;0|"
            "probesize;32768"
        )
    elif transport.lower() == "udp":
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;udp|"
            "stimeout;5000000|"
            "max_delay;500000|"
            "fflags;nobuffer|"
            "flags;low_delay"
        )
    else:
        os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)


@dataclass
class ReaderStats:
    connected: bool = False
    reconnects: int = 0
    read_failures: int = 0
    total_frames: int = 0
    last_frame_time: float = 0.0
    last_error: str = ""


class LatestFrameRTSPReader:
    """
    Reads RTSP frames in a background thread and keeps only the newest frame.
    This prevents the display loop from getting stuck behind old buffered frames.
    """

    def __init__(
        self,
        url: str,
        reconnect_after_seconds: float = 3.0,
        open_retry_delay_seconds: float = 1.0,
    ) -> None:
        self.url = url
        self.reconnect_after_seconds = reconnect_after_seconds
        self.open_retry_delay_seconds = open_retry_delay_seconds

        self._lock = threading.Lock()
        self._frame = None
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = ReaderStats()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="RTSPReader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest_frame(self):
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _open_capture(self):
        import cv2

        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        # These do not always work with FFmpeg/RTSP, but they are harmless.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        return cap

    def _run(self) -> None:
        while not self._stopped.is_set():
            cap = None
            try:
                cap = self._open_capture()

                if not cap.isOpened():
                    self.stats.connected = False
                    self.stats.last_error = "Could not open RTSP stream"
                    time.sleep(self.open_retry_delay_seconds)
                    self.stats.reconnects += 1
                    continue

                self.stats.connected = True
                self.stats.last_error = ""
                last_good = time.time()

                while not self._stopped.is_set():
                    ok, frame = cap.read()
                    now = time.time()

                    if not ok or frame is None:
                        self.stats.read_failures += 1

                        if now - last_good > self.reconnect_after_seconds:
                            self.stats.connected = False
                            self.stats.last_error = "No valid frames; reconnecting"
                            self.stats.reconnects += 1
                            break

                        time.sleep(0.01)
                        continue

                    last_good = now
                    with self._lock:
                        self._frame = frame

                    self.stats.connected = True
                    self.stats.total_frames += 1
                    self.stats.last_frame_time = now

            except Exception as exc:
                self.stats.connected = False
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                self.stats.reconnects += 1
                time.sleep(self.open_retry_delay_seconds)

            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass


def resize_max_width(frame, max_width: int):
    if max_width <= 0:
        return frame

    import cv2

    h, w = frame.shape[:2]
    if w <= max_width:
        return frame

    scale = max_width / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def apply_digital_zoom(frame, zoom: float, center_x: float, center_y: float):
    """
    Digital zoom by cropping around a normalized center and resizing back.
    center_x and center_y are in [0, 1].
    """
    if zoom <= 1.01:
        return frame

    import cv2

    h, w = frame.shape[:2]
    crop_w = max(1, int(w / zoom))
    crop_h = max(1, int(h / zoom))

    cx = int(center_x * w)
    cy = int(center_y * h)

    x1 = max(0, min(w - crop_w, cx - crop_w // 2))
    y1 = max(0, min(h - crop_h, cy - crop_h // 2))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def draw_overlay(
    frame,
    display_fps: float,
    stats: ReaderStats,
    zoom: float,
    center_x: float,
    center_y: float,
    source_label: str,
):
    import cv2

    h, w = frame.shape[:2]
    connected_text = "CONNECTED" if stats.connected else "RECONNECTING"
    age = time.time() - stats.last_frame_time if stats.last_frame_time else -1

    lines = [
        f"{source_label}",
        f"display_fps={display_fps:.1f} | size={w}x{h} | {connected_text}",
        f"frames={stats.total_frames} | failures={stats.read_failures} | reconnects={stats.reconnects}",
        f"zoom={zoom:.2f}x | center=({center_x:.2f}, {center_y:.2f}) | frame_age={age:.1f}s",
    ]

    if stats.last_error:
        lines.append(f"last_error={stats.last_error}")

    x, y = 12, 26
    line_h = 24

    # Background rectangle
    rect_h = line_h * len(lines) + 12
    cv2.rectangle(frame, (6, 6), (min(w - 6, 920), rect_h), (0, 0, 0), -1)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def save_snapshot(frame, snapshots_dir: Path) -> Path:
    import cv2

    snapshots_dir.mkdir(parents=True, exist_ok=True)
    name = time.strftime("rtsp_snapshot_%Y%m%d_%H%M%S.jpg")
    path = snapshots_dir / name
    cv2.imwrite(str(path), frame)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone RTSP camera viewer.")
    parser.add_argument("url", help="RTSP URL, for example rtsp://user:pass@ip:554/Streaming/channels/102")
    parser.add_argument(
        "--transport",
        choices=["tcp", "udp", "auto"],
        default="tcp",
        help="RTSP transport. Use tcp first. Default: tcp.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Maximum display width. Use 640 for weak PCs or unstable streams. Default: 960.",
    )
    parser.add_argument(
        "--window",
        default="RTSP Viewer",
        help="OpenCV window title.",
    )
    parser.add_argument(
        "--reconnect-after",
        type=float,
        default=3.0,
        help="Reconnect if no valid frame arrives for this many seconds. Default: 3.0.",
    )
    parser.add_argument(
        "--snapshots-dir",
        default="snapshots",
        help="Folder where snapshots are saved when pressing s.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_ffmpeg_transport(args.transport)

    import cv2

    reader = LatestFrameRTSPReader(
        args.url,
        reconnect_after_seconds=args.reconnect_after,
    )
    reader.start()

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    last_display_time = time.time()
    display_fps = 0.0
    zoom = 1.0
    center_x = 0.5
    center_y = 0.5
    source_label = "RTSP TCP" if args.transport == "tcp" else f"RTSP {args.transport.upper()}"

    try:
        while True:
            frame = reader.get_latest_frame()

            if frame is None:
                placeholder = 255 * __import__("numpy").ones((360, 640, 3), dtype="uint8")
                cv2.putText(
                    placeholder,
                    "Waiting for RTSP frames...",
                    (40, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    placeholder,
                    reader.stats.last_error or "Opening stream",
                    (40, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(args.window, placeholder)
                key = cv2.waitKeyEx(50)
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            frame = resize_max_width(frame, args.width)
            frame = apply_digital_zoom(frame, zoom, center_x, center_y)

            now = time.time()
            dt = max(1e-6, now - last_display_time)
            instant_fps = 1.0 / dt
            display_fps = instant_fps if display_fps <= 0 else (0.90 * display_fps + 0.10 * instant_fps)
            last_display_time = now

            frame = draw_overlay(
                frame,
                display_fps=display_fps,
                stats=reader.stats,
                zoom=zoom,
                center_x=center_x,
                center_y=center_y,
                source_label=source_label,
            )

            cv2.imshow(args.window, frame)
            key = cv2.waitKeyEx(1)

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("p"), ord("P")):
                saved = save_snapshot(frame, Path(args.snapshots_dir))
                print(f"Saved snapshot: {saved}")

            elif key in (ord("+"), ord("=")):
                zoom = min(8.0, zoom + 0.25)

            elif key in (ord("-"), ord("_")):
                zoom = max(1.0, zoom - 0.25)

            elif key in (ord("r"), ord("R")):
                zoom = 1.0
                center_x = 0.5
                center_y = 0.5

            # Arrow keys from cv2.waitKeyEx on Windows/Linux.
            elif key in (2424832, 65361):  # left
                center_x = max(0.0, center_x - 0.05)
            elif key in (2555904, 65363):  # right
                center_x = min(1.0, center_x + 0.05)
            elif key in (2490368, 65362):  # up
                center_y = max(0.0, center_y - 0.05)
            elif key in (2621440, 65364):  # down
                center_y = min(1.0, center_y + 0.05)

            # WASD works reliably for pan.
            elif key in (ord("a"), ord("A")):
                center_x = max(0.0, center_x - 0.05)
            elif key in (ord("d"), ord("D")):
                center_x = min(1.0, center_x + 0.05)
            elif key in (ord("w"), ord("W")):
                center_y = max(0.0, center_y - 0.05)
            elif key in (ord("s"), ord("S")):
                center_y = min(1.0, center_y + 0.05)

    finally:
        reader.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
