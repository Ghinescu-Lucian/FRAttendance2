"""
Simple automatic door / doorway zoom demo.

What it does:
  - Opens a webcam.
  - Searches for a tall rectangular door / doorway frame using edges + Hough vertical lines.
  - Draws the detected door on the full frame.
  - Digitally zooms the display onto that door so it is visible.

Install:
  py -m pip install opencv-python numpy

Run:
  py door_auto_zoom_simple.py

Optional:
  py door_auto_zoom_simple.py --camera 1
  py door_auto_zoom_simple.py --width 1280 --height 720

Keys:
  ESC or Q  exit
  D         toggle debug full-frame window
  R         reset zoom
  + / -     change zoom margin

Important:
  This is a classical computer-vision heuristic, not a trained AI door detector.
  It works best when the camera is static and the door/doorway has visible vertical edges.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


WINDOW_ZOOM = "Door auto zoom"
WINDOW_DEBUG = "Door detector debug"


@dataclass
class DoorCandidate:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    reason: str


class SmoothBox:
    """Small helper that smooths the zoom target so the view does not jump."""

    def __init__(self, alpha: float = 0.18):
        self.alpha = alpha
        self.box: Optional[Tuple[float, float, float, float]] = None

    def reset(self):
        self.box = None

    def update(self, target: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        tx1, ty1, tx2, ty2 = map(float, target)

        if self.box is None:
            self.box = (tx1, ty1, tx2, ty2)
        else:
            x1, y1, x2, y2 = self.box
            a = self.alpha
            self.box = (
                (1.0 - a) * x1 + a * tx1,
                (1.0 - a) * y1 + a * ty1,
                (1.0 - a) * x2 + a * tx2,
                (1.0 - a) * y2 + a * ty2,
            )

        sx1, sy1, sx2, sy2 = self.box
        return int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2))


def clamp_box(box: Tuple[int, int, int, int], frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(frame_w - 2, x1))
    y1 = max(0, min(frame_h - 2, y1))
    x2 = max(x1 + 1, min(frame_w - 1, x2))
    y2 = max(y1 + 1, min(frame_h - 1, y2))
    return x1, y1, x2, y2


def expand_box(
    box: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
    margin_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    mx = int(round(w * margin_ratio))
    my = int(round(h * margin_ratio))
    return clamp_box((x1 - mx, y1 - my, x2 + mx, y2 + my), frame_w, frame_h)


def make_crop_same_as_frame_aspect(
    box: Tuple[int, int, int, int],
    frame_w: int,
    frame_h: int,
) -> Tuple[int, int, int, int]:
    """
    Expand the crop around the detected door so resizing does not distort too much.
    The crop keeps the camera frame aspect ratio.
    """
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w = max(2, x2 - x1)
    crop_h = max(2, y2 - y1)

    frame_aspect = frame_w / float(frame_h)
    crop_aspect = crop_w / float(crop_h)

    if crop_aspect < frame_aspect:
        crop_w = int(round(crop_h * frame_aspect))
    else:
        crop_h = int(round(crop_w / frame_aspect))

    crop_w = min(crop_w, frame_w)
    crop_h = min(crop_h, frame_h)

    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))

    left = max(0, min(frame_w - crop_w, left))
    top = max(0, min(frame_h - crop_h, top))
    return left, top, left + crop_w, top + crop_h


def vertical_line_score(lines: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    """Score how many strong vertical Hough lines support a rectangle candidate."""
    if lines is None:
        return 0.0

    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    left_count = 0
    right_count = 0
    long_vertical_count = 0

    for line in lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = map(int, line)
        dx = abs(lx2 - lx1)
        dy = abs(ly2 - ly1)
        length = math.hypot(dx, dy)

        if length < 0.25 * bh:
            continue
        if dx > 0.18 * max(1, dy):
            continue

        lx = (lx1 + lx2) / 2.0
        ly_top = min(ly1, ly2)
        ly_bottom = max(ly1, ly2)

        # Must overlap vertically with the candidate.
        overlap = max(0, min(y2, ly_bottom) - max(y1, ly_top))
        if overlap < 0.25 * bh:
            continue

        long_vertical_count += 1

        if abs(lx - x1) < 0.22 * bw:
            left_count += 1
        if abs(lx - x2) < 0.22 * bw:
            right_count += 1

    support = min(left_count, 2) + min(right_count, 2) + min(long_vertical_count, 4) * 0.5
    return support


def contour_candidates(
    edges: np.ndarray,
    lines: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
) -> List[DoorCandidate]:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[DoorCandidate] = []
    frame_area = frame_w * frame_h

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 1 or h <= 1:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        box_area = w * h
        area_ratio = box_area / float(frame_area)
        aspect = h / float(w)
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / float(box_area)

        # Door / doorway rough shape filters.
        if h < 0.32 * frame_h:
            continue
        if w < 0.06 * frame_w:
            continue
        if area_ratio < 0.035 or area_ratio > 0.78:
            continue
        if aspect < 1.15 or aspect > 5.8:
            continue
        if rectangularity < 0.08:
            continue

        line_support = vertical_line_score(lines, (x1, y1, x2, y2))

        # Prefer tall centered-ish rectangles with vertical line support.
        height_score = h / float(frame_h)
        aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 3.0)
        rect_score = min(rectangularity * 2.0, 1.0)
        score = 4.0 * height_score + 2.0 * aspect_score + rect_score + line_support

        candidates.append(DoorCandidate((x1, y1, x2, y2), score, "contour+lines"))

    return candidates


def hough_pair_candidates(
    lines: Optional[np.ndarray],
    frame_w: int,
    frame_h: int,
) -> List[DoorCandidate]:
    """Fallback detector: find two tall vertical lines that look like a doorway frame."""
    if lines is None:
        return []

    verticals = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        length = math.hypot(dx, dy)

        if length < 0.28 * frame_h:
            continue
        if dx > 0.16 * max(1, dy):
            continue

        x = int(round((x1 + x2) / 2.0))
        top = min(y1, y2)
        bottom = max(y1, y2)
        verticals.append((x, top, bottom, length))

    candidates: List[DoorCandidate] = []

    for i in range(len(verticals)):
        lx, ltop, lbottom, llen = verticals[i]
        for j in range(i + 1, len(verticals)):
            rx, rtop, rbottom, rlen = verticals[j]
            if rx < lx:
                lx, rx = rx, lx
                ltop, rtop = rtop, ltop
                lbottom, rbottom = rbottom, lbottom
                llen, rlen = rlen, llen

            width = rx - lx
            if width < 0.07 * frame_w or width > 0.70 * frame_w:
                continue

            top = min(ltop, rtop)
            bottom = max(lbottom, rbottom)
            height = bottom - top
            if height < 0.32 * frame_h:
                continue

            aspect = height / float(width)
            if aspect < 1.15 or aspect > 6.0:
                continue

            overlap = max(0, min(lbottom, rbottom) - max(ltop, rtop))
            if overlap < 0.42 * height:
                continue

            area_ratio = (width * height) / float(frame_w * frame_h)
            if area_ratio < 0.025 or area_ratio > 0.80:
                continue

            height_score = height / float(frame_h)
            aspect_score = max(0.0, 1.0 - abs(aspect - 2.2) / 3.0)
            overlap_score = overlap / float(height)
            length_score = min((llen + rlen) / float(frame_h), 2.0)
            score = 4.0 * height_score + 2.0 * aspect_score + 2.0 * overlap_score + length_score

            candidates.append(DoorCandidate((lx, top, rx, bottom), score, "hough-pair"))

    return candidates


def find_door(frame: np.ndarray) -> Tuple[Optional[DoorCandidate], np.ndarray, Optional[np.ndarray]]:
    frame_h, frame_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Improve edge contrast when the room is dim.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    edges = cv2.Canny(gray, 55, 150)

    # Close small gaps in door-frame edges.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)

    min_line_length = int(max(80, 0.22 * frame_h))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=70,
        minLineLength=min_line_length,
        maxLineGap=25,
    )

    candidates = []
    candidates.extend(contour_candidates(closed, lines, frame_w, frame_h))
    candidates.extend(hough_pair_candidates(lines, frame_w, frame_h))

    if not candidates:
        return None, closed, lines

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0], closed, lines


def zoom_to_box(frame: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame.copy()
    return cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def draw_candidate(frame: np.ndarray, candidate: DoorCandidate, color=(0, 255, 255)) -> None:
    x1, y1, x2, y2 = candidate.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame,
        f"door? {candidate.score:.2f} {candidate.reason}",
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def draw_crop(frame: np.ndarray, crop_box: Tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = crop_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(
        frame,
        "zoom crop",
        (x1, min(frame.shape[0] - 10, y2 + 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 255),
        2,
    )


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple door detection and automatic digital zoom.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index. Default: 0")
    parser.add_argument("--width", type=int, default=960, help="Requested camera width. Default: 960")
    parser.add_argument("--height", type=int, default=540, help="Requested camera height. Default: 540")
    parser.add_argument("--margin", type=float, default=0.25, help="Extra space around detected door. Default: 0.25")
    parser.add_argument("--hold", type=int, default=35, help="Keep last door for N frames if temporarily lost. Default: 35")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = open_camera(args.camera, args.width, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print("Looking for a tall rectangular door / doorway frame...")
    print("Keys: ESC/Q exit | D debug | R reset | +/- zoom margin")

    smoother = SmoothBox(alpha=0.18)
    last_candidate: Optional[DoorCandidate] = None
    lost_frames = 9999
    show_debug = True
    margin = float(args.margin)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        frame_h, frame_w = frame.shape[:2]
        candidate, edge_debug, lines = find_door(frame)

        full_debug = frame.copy()

        if candidate is not None:
            last_candidate = candidate
            lost_frames = 0
        else:
            lost_frames += 1

        active_candidate = last_candidate if lost_frames <= args.hold else None

        if active_candidate is not None:
            door_box = expand_box(active_candidate.box, frame_w, frame_h, margin)
            crop_box = make_crop_same_as_frame_aspect(door_box, frame_w, frame_h)
            crop_box = smoother.update(crop_box)
            crop_box = clamp_box(crop_box, frame_w, frame_h)
            zoomed = zoom_to_box(frame, crop_box)

            draw_candidate(full_debug, active_candidate)
            draw_crop(full_debug, crop_box)

            status = "DETECTED" if candidate is not None else f"HOLDING LAST ({lost_frames}/{args.hold})"
            cv2.putText(
                zoomed,
                f"Door zoom: {status} | margin {margin:.2f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            smoother.reset()
            zoomed = frame.copy()
            cv2.putText(
                zoomed,
                "No door detected. Make sure the door frame is visible.",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            zoomed,
            "ESC/Q exit | D debug | R reset | +/- margin",
            (20, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_ZOOM, zoomed)

        if show_debug:
            # Draw vertical Hough lines to understand why a door is or isn't detected.
            if lines is not None:
                for line in lines[:, 0, :]:
                    x1, y1, x2, y2 = map(int, line)
                    dx = abs(x2 - x1)
                    dy = abs(y2 - y1)
                    if dy > 0 and dx <= 0.18 * dy:
                        cv2.line(full_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow(WINDOW_DEBUG, full_debug)
        else:
            try:
                cv2.destroyWindow(WINDOW_DEBUG)
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("d"), ord("D")):
            show_debug = not show_debug
        if key in (ord("r"), ord("R")):
            smoother.reset()
            last_candidate = None
            lost_frames = 9999
            print("Reset detector state.")
        if key in (ord("+"), ord("=")):
            margin = min(0.80, margin + 0.05)
            print(f"Zoom margin: {margin:.2f}")
        if key in (ord("-"), ord("_")):
            margin = max(0.00, margin - 0.05)
            print(f"Zoom margin: {margin:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
