import os
import time
from collections import defaultdict

import cv2
from simple_facerec import SimpleFacerec


# =========================
# Configuration
# =========================
CAMERA_INDEX = 1
ENCODINGS_DIR = "images/"
CAPTURE_DIR = "captures"
WINDOW_NAME = "Far Face Search Zoom Follow"

# Use the highest real resolution your webcam supports.
# This matters much more than digital zoom.
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

UNKNOWN_LABEL = "Unknown"

# Display zoom after a face target is found.
MIN_ZOOM = 1.0
MAX_ZOOM = 3.5
FACE_TARGET_HEIGHT = 0.38

CENTER_SMOOTHING = 0.22
ZOOM_SMOOTHING = 0.16

# Important for far faces:
# When the face is too small in the original frame, the program scans
# digitally zoomed crops and runs recognition on those enlarged crops.
SEARCH_ZOOM_LEVELS = [1.0, 1.8, 2.6, 3.4]
SEARCH_EVERY_N_FRAMES = 1

STABLE_FRAMES_REQUIRED = 4
PHOTO_COOLDOWN_SECONDS = 8
MAX_LOST_FRAMES_AFTER_LOCK = 20

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
            # Do not jump back to the full frame immediately.
            # Keep the last crop and slowly relax zoom.
            target_cx = self.center_x
            target_cy = self.center_y
            target_zoom = max(MIN_ZOOM, self.zoom * 0.98)
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


def make_search_crops(frame_w, frame_h):
    """
    Makes full-frame and zoomed search windows.
    Each crop is resized back to the full frame size before calling SimpleFacerec.
    This makes a far/small face appear larger to the detector.
    """
    crops = []

    for zoom in SEARCH_ZOOM_LEVELS:
        crop_w = int(frame_w / zoom)
        crop_h = int(frame_h / zoom)

        if zoom == 1.0:
            crops.append((0, 0, frame_w, frame_h, zoom, "full"))
            continue

        xs = [0, (frame_w - crop_w) // 2, frame_w - crop_w]
        ys = [0, (frame_h - crop_h) // 2, frame_h - crop_h]

        seen = set()
        for y in ys:
            for x in xs:
                x = max(0, min(frame_w - crop_w, x))
                y = max(0, min(frame_h - crop_h, y))
                key = (x, y, crop_w, crop_h)
                if key not in seen:
                    seen.add(key)
                    crops.append((x, y, crop_w, crop_h, zoom, f"search {zoom:.1f}x"))

    return crops


def detect_faces_with_search_zoom(sfr, frame):
    """
    Detect/recognize faces using:
    1. the original full frame
    2. multiple digitally zoomed search crops

    The returned face boxes are mapped back to original-frame coordinates.
    """
    frame_h, frame_w = frame.shape[:2]
    candidates = []

    for left, top, crop_w, crop_h, search_zoom, source in make_search_crops(frame_w, frame_h):
        crop = frame[top:top + crop_h, left:left + crop_w]

        if search_zoom == 1.0:
            recognition_input = crop
        else:
            recognition_input = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

        face_locations, face_names = sfr.detect_known_faces(recognition_input)

        if len(face_locations) == 0:
            continue

        for face_loc, name in zip(face_locations, face_names):
            bx1, by1, bx2, by2 = normalize_face_location(face_loc)

            if search_zoom == 1.0:
                original_box = (bx1, by1, bx2, by2)
            else:
                scale_x = crop_w / frame_w
                scale_y = crop_h / frame_h

                original_box = (
                    int(left + bx1 * scale_x),
                    int(top + by1 * scale_y),
                    int(left + bx2 * scale_x),
                    int(top + by2 * scale_y),
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

        # If full frame already found a known face, skip the expensive grid scan.
        if search_zoom == 1.0 and any(c["is_known"] for c in candidates):
            break

    return candidates


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


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

        # If recognition temporarily fails while walking, keep following the closest visible face.
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


def main():
    sfr = SimpleFacerec()
    sfr.load_encoding_images(ENCODINGS_DIR)

    # On Windows, CAP_DSHOW often starts webcams faster and respects resolution better.
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        frame_index += 1
        frame_h, frame_w = frame.shape[:2]

        candidates = []
        if frame_index % SEARCH_EVERY_N_FRAMES == 0:
            candidates = detect_faces_with_search_zoom(sfr, frame)

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
            print("Reset followed person.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
