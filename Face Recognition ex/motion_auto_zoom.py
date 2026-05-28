"""
motion_auto_zoom.py

Simple movement-based auto zoom.

Purpose:
- Do NOT try to recognize a door semantically.
- Watch the full camera frame for movement.
- When the door opens or a person enters, that movement becomes the target.
- Digitally zoom the displayed view toward the moving region.

Run:
    py -m pip install opencv-python numpy
    py motion_auto_zoom.py

Controls:
    ESC / Q       exit
    D             show/hide debug mask window
    B             reset/relearn background
    L             lock/unlock current zoom target
    + / =         zoom closer
    - / _         zoom out / include more context
    H             print help

Notes:
- This is digital zoom. It crops and enlarges the image. It cannot create real optical detail.
- Use a static camera. Moving curtains, reflections, monitors, or people already in the room can steal the target.
- Let the camera view sit still for 2-3 seconds before entering, so the background model stabilizes.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# =========================
# Configuration
# =========================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

MAIN_WINDOW = "Movement Auto Zoom"
FULL_WINDOW = "Full Camera + Movement Target"
DEBUG_WINDOW = "Motion Mask Debug"

# Lower = more sensitive, but more false movement.
MOTION_THRESHOLD = 200

# Smaller values detect smaller movements.
MIN_MOTION_AREA_RATIO = 0.004

# Ignore huge global changes like exposure flashes if above this.
MAX_MOTION_AREA_RATIO = 0.65

# Resize width used only for motion detection speed.
MOTION_PROCESS_WIDTH = 640

# Background subtractor settings.
BG_HISTORY = 120
BG_VAR_THRESHOLD = 28
BG_DETECT_SHADOWS = True

# Morphological cleanup.
MORPH_KERNEL_SIZE = 5
DILATE_ITERATIONS = 3

# Zoom behavior.
INITIAL_MARGIN_RATIO = 0.55
MIN_MARGIN_RATIO = 0.05
MAX_MARGIN_RATIO = 2.50
MARGIN_STEP = 0.08

# Minimum crop prevents over-zooming into a few pixels.
MIN_CROP_WIDTH_RATIO = 0.16
MIN_CROP_HEIGHT_RATIO = 0.16

# Maximum crop prevents "zoom" from becoming almost the whole frame when target is too large.
MAX_CROP_WIDTH_RATIO = 1.00
MAX_CROP_HEIGHT_RATIO = 1.00

# Target smoothing. Higher = follows faster. Lower = steadier.
CENTER_SMOOTHING = 0.22
SIZE_SMOOTHING = 0.16

# How many frames with motion before target is trusted.
STABLE_MOTION_FRAMES_REQUIRED = 3

# How long to keep zoom on last movement after movement stops.
HOLD_TARGET_SECONDS = 4.0

# If true, when target is found, keep it locked until L/B/R.
LOCK_AFTER_FIRST_TARGET = False

# Prefer vertical/tall motion, useful for doors and people.
PREFER_TALL_TARGETS = True

# Draw settings.
TARGET_COLOR = (0, 255, 0)
MOTION_COLOR = (0, 200, 255)
CROP_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
WARN_COLOR = (0, 180, 255)


@dataclass
class MotionTarget:
    box: Tuple[int, int, int, int]
    score: float
    area_ratio: float
    timestamp: float


class SmoothZoom:
    def __init__(self):
        self.center_x = None
        self.center_y = None
        self.crop_w = None
        self.crop_h = None

    def reset(self):
        self.center_x = None
        self.center_y = None
        self.crop_w = None
        self.crop_h = None

    def update(self, frame_w, frame_h, target_box: Optional[Tuple[int, int, int, int]], margin_ratio: float):
        if target_box is None:
            if self.center_x is None:
                self.center_x = frame_w / 2.0
                self.center_y = frame_h / 2.0
                self.crop_w = frame_w
                self.crop_h = frame_h
            return self.get_crop(frame_w, frame_h)

        x1, y1, x2, y2 = target_box
        target_cx = (x1 + x2) / 2.0
        target_cy = (y1 + y2) / 2.0
        target_w = max(1, x2 - x1)
        target_h = max(1, y2 - y1)

        desired_w = target_w * (1.0 + 2.0 * margin_ratio)
        desired_h = target_h * (1.0 + 2.0 * margin_ratio)

        min_w = frame_w * MIN_CROP_WIDTH_RATIO
        min_h = frame_h * MIN_CROP_HEIGHT_RATIO
        max_w = frame_w * MAX_CROP_WIDTH_RATIO
        max_h = frame_h * MAX_CROP_HEIGHT_RATIO

        desired_w = max(min_w, min(max_w, desired_w))
        desired_h = max(min_h, min(max_h, desired_h))

        # Preserve camera aspect ratio to avoid stretching.
        frame_aspect = frame_w / float(frame_h)
        desired_aspect = desired_w / float(desired_h)

        if desired_aspect > frame_aspect:
            desired_h = desired_w / frame_aspect
        else:
            desired_w = desired_h * frame_aspect

        desired_w = max(min_w, min(frame_w, desired_w))
        desired_h = max(min_h, min(frame_h, desired_h))

        if self.center_x is None:
            self.center_x = target_cx
            self.center_y = target_cy
            self.crop_w = desired_w
            self.crop_h = desired_h
        else:
            self.center_x = (1.0 - CENTER_SMOOTHING) * self.center_x + CENTER_SMOOTHING * target_cx
            self.center_y = (1.0 - CENTER_SMOOTHING) * self.center_y + CENTER_SMOOTHING * target_cy
            self.crop_w = (1.0 - SIZE_SMOOTHING) * self.crop_w + SIZE_SMOOTHING * desired_w
            self.crop_h = (1.0 - SIZE_SMOOTHING) * self.crop_h + SIZE_SMOOTHING * desired_h

        return self.get_crop(frame_w, frame_h)

    def get_crop(self, frame_w, frame_h):
        crop_w = int(round(max(1, min(frame_w, self.crop_w or frame_w))))
        crop_h = int(round(max(1, min(frame_h, self.crop_h or frame_h))))

        cx = float(self.center_x if self.center_x is not None else frame_w / 2.0)
        cy = float(self.center_y if self.center_y is not None else frame_h / 2.0)

        left = int(round(cx - crop_w / 2.0))
        top = int(round(cy - crop_h / 2.0))

        left = max(0, min(frame_w - crop_w, left))
        top = max(0, min(frame_h - crop_h, top))

        return left, top, left + crop_w, top + crop_h


def clamp(value, low, high):
    return max(low, min(high, value))


def print_help():
    print(__doc__)


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


def resize_for_motion(frame):
    h, w = frame.shape[:2]
    if w <= MOTION_PROCESS_WIDTH:
        return frame, 1.0, 1.0

    scale = MOTION_PROCESS_WIDTH / float(w)
    new_h = int(round(h * scale))
    resized = cv2.resize(frame, (MOTION_PROCESS_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    scale_back_x = w / float(MOTION_PROCESS_WIDTH)
    scale_back_y = h / float(new_h)
    return resized, scale_back_x, scale_back_y


def create_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_VAR_THRESHOLD,
        detectShadows=BG_DETECT_SHADOWS,
    )


def clean_motion_mask(fg_mask):
    # MOG2 shadows are usually 127. Keep only strong foreground.
    _, mask = cv2.threshold(fg_mask, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=DILATE_ITERATIONS)

    return mask


def merge_boxes(boxes):
    if not boxes:
        return None

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    return x1, y1, x2, y2


def score_motion_box(box, frame_w, frame_h, contour_area):
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    area_ratio = contour_area / float(frame_w * frame_h)

    score = contour_area

    if PREFER_TALL_TARGETS:
        aspect_tall_bonus = clamp(h / float(w), 0.5, 3.0)
        score *= aspect_tall_bonus

    # Slightly prefer movement not at the extreme top/bottom edges.
    cy = (y1 + y2) / 2.0
    center_y_weight = 1.0 - 0.25 * abs((cy / frame_h) - 0.5)
    score *= center_y_weight

    return score, area_ratio


def find_motion_target(frame, bg_subtractor):
    frame_h, frame_w = frame.shape[:2]

    small, scale_back_x, scale_back_y = resize_for_motion(frame)
    small_h, small_w = small.shape[:2]

    blurred = cv2.GaussianBlur(small, (5, 5), 0)
    fg_mask = bg_subtractor.apply(blurred)
    mask = clean_motion_mask(fg_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    debug_boxes_small = []

    for contour in contours:
        contour_area_small = cv2.contourArea(contour)
        if contour_area_small <= 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        sx1, sy1, sx2, sy2 = x, y, x + w, y + h

        # Convert area from small scale to original-ish scale.
        contour_area = contour_area_small * scale_back_x * scale_back_y
        area_ratio = contour_area / float(frame_w * frame_h)

        if area_ratio < MIN_MOTION_AREA_RATIO:
            continue
        if area_ratio > MAX_MOTION_AREA_RATIO:
            continue

        ox1 = int(round(sx1 * scale_back_x))
        oy1 = int(round(sy1 * scale_back_y))
        ox2 = int(round(sx2 * scale_back_x))
        oy2 = int(round(sy2 * scale_back_y))

        box = (
            clamp(ox1, 0, frame_w - 1),
            clamp(oy1, 0, frame_h - 1),
            clamp(ox2, 0, frame_w - 1),
            clamp(oy2, 0, frame_h - 1),
        )

        score, area_ratio = score_motion_box(box, frame_w, frame_h, contour_area)

        candidates.append((box, score, area_ratio))
        debug_boxes_small.append((sx1, sy1, sx2, sy2))

    if not candidates:
        return None, mask, []

    # Keep strongest few nearby/visible movement components.
    candidates.sort(key=lambda item: item[1], reverse=True)
    top = candidates[:4]

    # If several components move at once because a person opens the door and enters,
    # union them so zoom covers the whole event.
    boxes = [item[0] for item in top]
    merged = merge_boxes(boxes)
    if merged is None:
        return None, mask, []

    total_score = sum(item[1] for item in top)
    total_area_ratio = sum(item[2] for item in top)

    target = MotionTarget(
        box=merged,
        score=total_score,
        area_ratio=total_area_ratio,
        timestamp=time.time(),
    )

    return target, mask, [item[0] for item in candidates]


def crop_and_zoom(frame, crop_box):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = crop_box

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return frame.copy()

    return cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def draw_text(img, text, x, y, scale=0.65, color=TEXT_COLOR, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_full_debug(frame, motion_boxes, current_target, crop_box, locked, margin_ratio):
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    for box in motion_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), MOTION_COLOR, 1)

    if current_target is not None:
        x1, y1, x2, y2 = current_target.box
        cv2.rectangle(out, (x1, y1), (x2, y2), TARGET_COLOR, 3)
        draw_text(out, "MOVEMENT TARGET", x1, max(25, y1 - 10), 0.65, TARGET_COLOR)

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
        cv2.rectangle(out, (x1, y1), (x2, y2), CROP_COLOR, 2)
        draw_text(out, "ZOOM CROP", x1, min(frame_h - 15, y2 + 25), 0.60, CROP_COLOR)

    status = f"Movement auto-zoom | margin={margin_ratio:.2f}"
    if locked:
        status += " | LOCKED"
    draw_text(out, status, 20, 35, 0.75)

    draw_text(out, "B reset bg | L lock | D debug | +/- zoom | ESC/Q exit", 20, frame_h - 20, 0.62)

    return out


def main():
    global INITIAL_MARGIN_RATIO

    print_help()

    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Could not read first camera frame")

    frame_h, frame_w = frame.shape[:2]
    print(f"[INFO] Camera resolution: {frame_w}x{frame_h}")
    print("[INFO] Let the camera view stay still for 2-3 seconds, then move/open the door.")

    bg_subtractor = create_background_subtractor()
    zoomer = SmoothZoom()

    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(FULL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, min(frame_w, 1000), int(min(frame_w, 1000) * frame_h / frame_w))
    cv2.resizeWindow(FULL_WINDOW, min(frame_w, 1000), int(min(frame_w, 1000) * frame_h / frame_w))

    show_debug = False
    locked = False
    margin_ratio = INITIAL_MARGIN_RATIO

    current_target: Optional[MotionTarget] = None
    last_valid_target: Optional[MotionTarget] = None
    stable_motion_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Could not read frame from camera.")
            break

        now = time.time()
        frame_h, frame_w = frame.shape[:2]

        detected_target, mask, motion_boxes = find_motion_target(frame, bg_subtractor)

        if not locked:
            if detected_target is not None:
                stable_motion_frames += 1

                if stable_motion_frames >= STABLE_MOTION_FRAMES_REQUIRED:
                    current_target = detected_target
                    last_valid_target = detected_target

                    if LOCK_AFTER_FIRST_TARGET:
                        locked = True
                        print("[OK] Movement target found and locked.")
            else:
                stable_motion_frames = 0

                if last_valid_target is not None and now - last_valid_target.timestamp <= HOLD_TARGET_SECONDS:
                    current_target = last_valid_target
                else:
                    current_target = None
        else:
            # Keep current target frozen.
            pass

        target_box = current_target.box if current_target is not None else None
        crop_box = zoomer.update(frame_w, frame_h, target_box, margin_ratio)
        zoomed = crop_and_zoom(frame, crop_box)

        if current_target is None:
            draw_text(zoomed, "Waiting for movement...", 20, 40, 0.85, WARN_COLOR)
        else:
            draw_text(zoomed, "ZOOMED ON MOVEMENT", 20, 40, 0.85, TARGET_COLOR)
            draw_text(zoomed, f"motion area={current_target.area_ratio:.3f} margin={margin_ratio:.2f}", 20, 75, 0.65)

        full_debug = draw_full_debug(frame, motion_boxes, current_target, crop_box, locked, margin_ratio)

        cv2.imshow(MAIN_WINDOW, zoomed)
        cv2.imshow(FULL_WINDOW, full_debug)

        if show_debug:
            cv2.imshow(DEBUG_WINDOW, mask)
        else:
            try:
                cv2.destroyWindow(DEBUG_WINDOW)
            except Exception:
                pass

        key = cv2.waitKeyEx(1)

        if key in (27, ord("q"), ord("Q")):
            break

        elif key in (ord("h"), ord("H")):
            print_help()

        elif key in (ord("d"), ord("D")):
            show_debug = not show_debug
            print(f"[OK] Debug mask window: {'ON' if show_debug else 'OFF'}")

        elif key in (ord("b"), ord("B"), ord("r"), ord("R")):
            bg_subtractor = create_background_subtractor()
            current_target = None
            last_valid_target = None
            stable_motion_frames = 0
            locked = False
            zoomer.reset()
            print("[OK] Background reset. Keep the camera still for 2-3 seconds.")

        elif key in (ord("l"), ord("L")):
            locked = not locked
            print(f"[OK] Target lock: {'ON' if locked else 'OFF'}")

        elif key in (ord("+"), ord("=")):
            margin_ratio = clamp(margin_ratio - MARGIN_STEP, MIN_MARGIN_RATIO, MAX_MARGIN_RATIO)
            print(f"[OK] Zoom closer. margin={margin_ratio:.2f}")

        elif key in (ord("-"), ord("_")):
            margin_ratio = clamp(margin_ratio + MARGIN_STEP, MIN_MARGIN_RATIO, MAX_MARGIN_RATIO)
            print(f"[OK] Zoom out. margin={margin_ratio:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
