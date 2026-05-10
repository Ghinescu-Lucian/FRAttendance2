import os
import time
from collections import defaultdict

import cv2
from simple_facerec import SimpleFacerec


# ============================================================
# FAST FAR-FACE SEARCH + AUTO-ZOOM + FOLLOW
# ============================================================
# Goal:
# 1. Search for faces even when the person is far from the camera.
# 2. Zoom/crop around the face so recognition has a better chance.
# 3. When a known person is recognized for several frames, save a photo.
# 4. Keep following that recognized person.
#
# Main speed improvement:
# - It does NOT scan every crop on every frame.
# - It searches around the last known face first.
# - It performs a full 3x3 grid search only sometimes.
# ============================================================


# =========================
# Configuration
# =========================
CAMERA_INDEX = 1          # Change to 0 if your webcam is the default camera
ENCODINGS_DIR = "images/"
CAPTURE_DIR = "captures"
WINDOW_NAME = "Fast Far Face Search Zoom Follow"

# For speed, start with 1280x720.
# If your PC is fast and your camera supports it, try 1920x1080.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

UNKNOWN_LABEL = "Unknown"

# Final display zoom after a face target is found.
MIN_ZOOM = 1.0
MAX_ZOOM = 3.5
FACE_TARGET_HEIGHT = 0.38

CENTER_SMOOTHING = 0.22
ZOOM_SMOOTHING = 0.16

# Recognition input size for zoom-search crops.
# Lower = faster, but weaker far-face detection.
# Good values: 800, 960, 1280
SEARCH_INPUT_WIDTH = 960

# Search behavior.
# Fast search is done every N frames.
# Full-grid search is more expensive, so it runs less often.
FAST_SEARCH_EVERY_N_FRAMES = 2
FULL_GRID_SEARCH_EVERY_N_FRAMES = 12

# Search zooms.
# More levels = better far detection but slower.
SEARCH_ZOOM_LEVELS_FAST = [1.0, 2.4, 3.4]
SEARCH_ZOOM_LEVELS_FULL = [1.0, 1.8, 2.6, 3.4]

# Tracking/follow behavior.
STABLE_FRAMES_REQUIRED = 4
PHOTO_COOLDOWN_SECONDS = 8
MAX_LOST_FRAMES_AFTER_LOCK = 24

SAVE_PHOTO_WHEN_KNOWN_STABLE = True
DRAW_DEBUG_SEARCH_SOURCE = False


os.makedirs(CAPTURE_DIR, exist_ok=True)


class FaceCenterZoomer:
    def __init__(self):
        self.center_x = None
        self.center_y = None
        self.zoom = MIN_ZOOM

    def _clamp_center(self, cx, cy, frame_w, frame_h, zoom):
        crop_w = frame_w / zoom
        crop_h = frame_h / zoom
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        cx = max(half_w, min(frame_w - half_w, cx))
        cy = max(half_h, min(frame_h - half_h, cy))
        return cx, cy

    def update(self, frame_shape, face_box=None):
        frame_h, frame_w = frame_shape[:2]

        if self.center_x is None or self.center_y is None:
            self.center_x = frame_w / 2.0
            self.center_y = frame_h / 2.0

        if face_box is None:
            # Do not jump back to full frame immediately.
            target_cx = self.center_x
            target_cy = self.center_y
            target_zoom = max(MIN_ZOOM, self.zoom * 0.985)
        else:
            x1, y1, x2, y2 = face_box
            face_h = max(1, y2 - y1)

            target_cx = (x1 + x2) / 2.0
            target_cy = (y1 + y2) / 2.0
            target_zoom = (FACE_TARGET_HEIGHT * frame_h) / face_h
            target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, target_zoom))

        self.center_x = (1.0 - CENTER_SMOOTHING) * self.center_x + CENTER_SMOOTHING * target_cx
        self.center_y = (1.0 - CENTER_SMOOTHING) * self.center_y + CENTER_SMOOTHING * target_cy
        self.zoom = (1.0 - ZOOM_SMOOTHING) * self.zoom + ZOOM_SMOOTHING * target_zoom
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))

        self.center_x, self.center_y = self._clamp_center(
            self.center_x, self.center_y, frame_w, frame_h, self.zoom
        )

        return self.get_crop_params(frame_w, frame_h)

    def get_crop_params(self, frame_w, frame_h):
        crop_w = int(frame_w / self.zoom)
        crop_h = int(frame_h / self.zoom)

        left = int(round(self.center_x - crop_w / 2.0))
        top = int(round(self.center_y - crop_h / 2.0))

        left = max(0, min(frame_w - crop_w, left))
        top = max(0, min(frame_h - crop_h, top))

        return left, top, crop_w, crop_h

    @staticmethod
    def apply_zoom(frame, crop_params):
        frame_h, frame_w = frame.shape[:2]
        left, top, crop_w, crop_h = crop_params
        crop = frame[top:top + crop_h, left:left + crop_w]
        return cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


def normalize_face_location(face_loc):
    """
    Supports both coordinate variants:
    - [top, right, bottom, left]
    - [top, left, bottom, right]

    Returns x1, y1, x2, y2.
    """
    y1 = int(face_loc[0])
    xa = int(face_loc[1])
    y2 = int(face_loc[2])
    xb = int(face_loc[3])

    x1 = min(xa, xb)
    x2 = max(xa, xb)
    y1, y2 = min(y1, y2), max(y1, y2)

    return x1, y1, x2, y2


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def make_crop_around_box(frame_w, frame_h, zoom, reference_box):
    crop_w = int(frame_w / zoom)
    crop_h = int(frame_h / zoom)

    cx, cy = box_center(reference_box)

    left = int(round(cx - crop_w / 2.0))
    top = int(round(cy - crop_h / 2.0))

    left = max(0, min(frame_w - crop_w, left))
    top = max(0, min(frame_h - crop_h, top))

    return left, top, crop_w, crop_h


def make_search_crops(frame_w, frame_h, zoom_levels, full_grid=False, last_box=None):
    """
    Creates search crops.

    Fast mode:
    - full frame
    - center crops
    - crops around the last known face

    Full-grid mode:
    - full frame
    - 3x3 grid for each zoom level
    """
    crops = []
    seen = set()

    def add_crop(left, top, crop_w, crop_h, zoom, source):
        left = max(0, min(frame_w - crop_w, int(left)))
        top = max(0, min(frame_h - crop_h, int(top)))
        key = (left, top, crop_w, crop_h)
        if key not in seen:
            seen.add(key)
            crops.append((left, top, crop_w, crop_h, zoom, source))

    for zoom in zoom_levels:
        crop_w = int(frame_w / zoom)
        crop_h = int(frame_h / zoom)

        if zoom == 1.0:
            add_crop(0, 0, frame_w, frame_h, zoom, "full")
            continue

        if last_box is not None:
            left, top, crop_w, crop_h = make_crop_around_box(frame_w, frame_h, zoom, last_box)
            add_crop(left, top, crop_w, crop_h, zoom, f"last-box {zoom:.1f}x")

        # Always check the center because most people stand in front of the camera.
        add_crop(
            (frame_w - crop_w) // 2,
            (frame_h - crop_h) // 2,
            crop_w,
            crop_h,
            zoom,
            f"center {zoom:.1f}x",
        )

        if full_grid:
            xs = [0, (frame_w - crop_w) // 2, frame_w - crop_w]
            ys = [0, (frame_h - crop_h) // 2, frame_h - crop_h]

            for y in ys:
                for x in xs:
                    add_crop(x, y, crop_w, crop_h, zoom, f"grid {zoom:.1f}x")

    return crops


def resize_for_recognition(image):
    h, w = image.shape[:2]

    if w <= SEARCH_INPUT_WIDTH:
        return image, 1.0, 1.0

    scale = SEARCH_INPUT_WIDTH / float(w)
    new_w = SEARCH_INPUT_WIDTH
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    scale_x = w / float(new_w)
    scale_y = h / float(new_h)

    return resized, scale_x, scale_y


def detect_faces_with_search_zoom(
    sfr,
    frame,
    full_grid=False,
    last_box=None,
    locked_name=None,
):
    """
    Detect/recognize faces using fewer crops than the slow version.

    The returned face boxes are mapped back to the original camera frame.
    """
    frame_h, frame_w = frame.shape[:2]
    candidates = []

    zoom_levels = SEARCH_ZOOM_LEVELS_FULL if full_grid else SEARCH_ZOOM_LEVELS_FAST

    search_crops = make_search_crops(
        frame_w,
        frame_h,
        zoom_levels=zoom_levels,
        full_grid=full_grid,
        last_box=last_box,
    )

    for left, top, crop_w, crop_h, search_zoom, source in search_crops:
        crop = frame[top:top + crop_h, left:left + crop_w]

        if search_zoom == 1.0:
            recognition_input = crop
            crop_to_recognition_scale_x = 1.0
            crop_to_recognition_scale_y = 1.0
        else:
            # Enlarge the crop to make a far-away face more detectable.
            enlarged = cv2.resize(
                crop,
                (frame_w, frame_h),
                interpolation=cv2.INTER_LINEAR,
            )

            recognition_input, resize_back_x, resize_back_y = resize_for_recognition(enlarged)

            # Final mapping:
            # recognition_input -> enlarged -> original crop -> original frame
            crop_to_recognition_scale_x = (crop_w / frame_w) * resize_back_x
            crop_to_recognition_scale_y = (crop_h / frame_h) * resize_back_y

        face_locations, face_names = sfr.detect_known_faces(recognition_input)

        if len(face_locations) == 0:
            continue

        for face_loc, name in zip(face_locations, face_names):
            bx1, by1, bx2, by2 = normalize_face_location(face_loc)

            if search_zoom == 1.0:
                # If full frame was resized, map back to full frame size.
                if recognition_input.shape[1] != crop_w:
                    resized_full, sx, sy = resize_for_recognition(crop)
                    # Normally not used because full frame is not resized here.
                    pass

                original_box = (bx1, by1, bx2, by2)
            else:
                original_box = (
                    int(left + bx1 * crop_to_recognition_scale_x),
                    int(top + by1 * crop_to_recognition_scale_y),
                    int(left + bx2 * crop_to_recognition_scale_x),
                    int(top + by2 * crop_to_recognition_scale_y),
                )

            x1, y1, x2, y2 = original_box
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            area = box_w * box_h

            is_known = bool(name) and name != UNKNOWN_LABEL

            candidates.append({
                "box": original_box,
                "name": name,
                "is_known": is_known,
                "area": area,
                "source": source,
                "search_zoom": search_zoom,
            })

            # Speed optimization:
            # If we are already following a person and found the same name,
            # stop scanning this frame.
            if locked_name and name == locked_name:
                return candidates

        # Speed optimization:
        # If normal full frame already found a known face, stop early.
        if source == "full" and any(c["is_known"] for c in candidates):
            break

    return candidates


def choose_target_face(candidates, frame_w, frame_h, locked_name=None, last_box=None):
    if not candidates:
        return None

    frame_center_box = (frame_w // 2, frame_h // 2, frame_w // 2 + 1, frame_h // 2 + 1)

    if locked_name:
        same_name = [c for c in candidates if c["name"] == locked_name]

        if same_name:
            reference = last_box if last_box is not None else frame_center_box
            same_name.sort(key=lambda c: center_distance(c["box"], reference))
            return same_name[0]

        # If recognition temporarily fails while walking,
        # follow the closest visible face to the last known face position.
        if last_box is not None:
            candidates.sort(key=lambda c: center_distance(c["box"], last_box))
            return candidates[0]

    known_faces = [c for c in candidates if c["is_known"]]

    if known_faces:
        known_faces.sort(key=lambda c: (-c["area"], center_distance(c["box"], frame_center_box)))
        return known_faces[0]

    candidates.sort(key=lambda c: (-c["area"], center_distance(c["box"], frame_center_box)))
    return candidates[0]


def transform_box_to_zoomed_view(box, crop_params, output_w, output_h):
    x1, y1, x2, y2 = box
    left, top, crop_w, crop_h = crop_params

    sx = output_w / crop_w
    sy = output_h / crop_h

    zx1 = int((x1 - left) * sx)
    zy1 = int((y1 - top) * sy)
    zx2 = int((x2 - left) * sx)
    zy2 = int((y2 - top) * sy)

    zx1 = max(0, min(output_w - 1, zx1))
    zy1 = max(0, min(output_h - 1, zy1))
    zx2 = max(0, min(output_w - 1, zx2))
    zy2 = max(0, min(output_h - 1, zy2))

    return zx1, zy1, zx2, zy2


def crop_face_from_frame(frame, face_box, margin=0.45):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = face_box
    w = x2 - x1
    h = y2 - y1

    pad_x = int(w * margin)
    pad_y = int(h * margin)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)

    return frame[y1:y2, x1:x2]


def save_person_photo(name, zoomed_frame, original_frame, face_box):
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    person_path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}_zoomed.jpg")
    face_path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}_face.jpg")

    cv2.imwrite(person_path, zoomed_frame)

    face_crop = crop_face_from_frame(original_frame, face_box)
    if face_crop.size > 0:
        cv2.imwrite(face_path, face_crop)

    return person_path, face_path


def open_camera():
    # On Windows, CAP_DSHOW often starts webcams faster and respects resolution better.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    return cap


def main():
    sfr = SimpleFacerec()
    sfr.load_encoding_images(ENCODINGS_DIR)

    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")

    zoomer = FaceCenterZoomer()

    locked_name = None
    last_box = None
    lost_frames = 0

    stable_name = None
    stable_count = 0
    last_capture_time_by_name = defaultdict(lambda: 0.0)

    frame_index = 0
    last_candidates = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        frame_index += 1
        frame_h, frame_w = frame.shape[:2]

        should_fast_search = frame_index % FAST_SEARCH_EVERY_N_FRAMES == 0
        should_full_grid_search = frame_index % FULL_GRID_SEARCH_EVERY_N_FRAMES == 0

        candidates = last_candidates

        if should_fast_search or should_full_grid_search:
            candidates = detect_faces_with_search_zoom(
                sfr,
                frame,
                full_grid=should_full_grid_search,
                last_box=last_box,
                locked_name=locked_name,
            )

            last_candidates = candidates

        target_face = choose_target_face(
            candidates,
            frame_w,
            frame_h,
            locked_name=locked_name,
            last_box=last_box,
        )

        if target_face is not None:
            last_box = target_face["box"]
            lost_frames = 0
        else:
            lost_frames += 1

            if locked_name and last_box is not None and lost_frames <= MAX_LOST_FRAMES_AFTER_LOCK:
                # Keep the last known crop for a short time so the view does not jump while walking.
                target_face = {
                    "box": last_box,
                    "name": locked_name,
                    "is_known": True,
                    "source": "last known",
                    "search_zoom": 0,
                }
            else:
                if lost_frames > MAX_LOST_FRAMES_AFTER_LOCK:
                    locked_name = None
                    last_box = None
                    last_candidates = []

                target_face = None

        face_box = target_face["box"] if target_face else None
        crop_params = zoomer.update(frame.shape, face_box=face_box)
        display_frame = FaceCenterZoomer.apply_zoom(frame, crop_params)

        if target_face is not None:
            name = target_face["name"]
            is_known = target_face["is_known"]

            if is_known:
                if name == stable_name:
                    stable_count += 1
                else:
                    stable_name = name
                    stable_count = 1
            else:
                stable_name = None
                stable_count = 0

            if is_known and stable_count >= STABLE_FRAMES_REQUIRED:
                locked_name = name

                if SAVE_PHOTO_WHEN_KNOWN_STABLE:
                    now = time.time()

                    if now - last_capture_time_by_name[name] >= PHOTO_COOLDOWN_SECONDS:
                        clean_zoomed = FaceCenterZoomer.apply_zoom(frame, crop_params)
                        person_path, face_path = save_person_photo(name, clean_zoomed, frame, face_box)
                        last_capture_time_by_name[name] = now

                        print(f"Saved photo for {name}: {person_path}")
                        print(f"Saved face crop for {name}: {face_path}")

            x1, y1, x2, y2 = transform_box_to_zoomed_view(face_box, crop_params, frame_w, frame_h)
            color = (0, 180, 0) if is_known else (0, 0, 200)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                display_frame,
                name,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                color,
                2,
            )

            if DRAW_DEBUG_SEARCH_SOURCE:
                cv2.putText(
                    display_frame,
                    f"source: {target_face.get('source', '')}",
                    (x1, min(frame_h - 15, y2 + 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
        else:
            stable_name = None
            stable_count = 0

        status = f"Zoom: {zoomer.zoom:.2f}x"

        if locked_name:
            status += f" | FOLLOWING {locked_name}"
        elif target_face is not None:
            status += " | Centering/searching face"
        else:
            status += " | Searching far face"

        cv2.putText(
            display_frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            "ESC = exit | R = reset followed person",
            (20, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if key in (ord("r"), ord("R")):
            locked_name = None
            last_box = None
            lost_frames = 0
            stable_name = None
            stable_count = 0
            last_candidates = []
            print("Reset followed person.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
