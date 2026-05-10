import os
import time
from collections import defaultdict

import cv2
from simple_facerec import SimpleFacerec


# =========================
# Configuration
# =========================
CAMERA_INDEX = 1          # Change to 0 if your webcam is the default camera
ENCODINGS_DIR = "images/"
CAPTURE_DIR = "captures"
WINDOW_NAME = "Face Center Auto Zoom"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

UNKNOWN_LABEL = "Unknown"

MIN_ZOOM = 1.0
MAX_ZOOM = 2.2            # Do not make this too high; digital zoom reduces quality
FACE_TARGET_HEIGHT = 0.34 # Face should occupy about 34% of the visible frame height

CENTER_SMOOTHING = 0.18   # Lower = smoother, higher = faster movement
ZOOM_SMOOTHING = 0.12

STABLE_FRAMES_REQUIRED = 6
PHOTO_COOLDOWN_SECONDS = 8
DRAW_ONLY_PRIMARY_FACE = True
SAVE_PHOTO_WHEN_KNOWN_STABLE = True


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
            target_cx = frame_w / 2.0
            target_cy = frame_h / 2.0
            target_zoom = MIN_ZOOM
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


def choose_primary_face(face_locations, face_names, frame_w, frame_h):
    """
    Select one face only.
    Priority:
    1. known recognized face
    2. largest face
    3. face closest to frame center
    """
    if len(face_locations) == 0:
        return None

    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0
    candidates = []

    for face_loc, name in zip(face_locations, face_names):
        box = normalize_face_location(face_loc)
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_distance = ((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5
        is_known = bool(name) and name != UNKNOWN_LABEL

        candidates.append({
            "box": box,
            "name": name,
            "is_known": is_known,
            "area": area,
            "center_distance": center_distance,
        })

    # Known faces first, then larger faces, then more central faces.
    candidates.sort(key=lambda f: (not f["is_known"], -f["area"], f["center_distance"]))
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


def save_person_photo(name, zoomed_clean_frame):
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}.jpg")
    cv2.imwrite(path, zoomed_clean_frame)
    return path


def main():
    sfr = SimpleFacerec()
    sfr.load_encoding_images(ENCODINGS_DIR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    zoomer = FaceCenterZoomer()
    last_stable_name = None
    stable_count = 0
    last_capture_time_by_name = defaultdict(lambda: 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        frame_h, frame_w = frame.shape[:2]

        # Only face detection/recognition is used for targeting.
        # No hand/motion/person boxes are used.
        face_locations, face_names = sfr.detect_known_faces(frame)
        primary_face = choose_primary_face(face_locations, face_names, frame_w, frame_h)

        face_box = primary_face["box"] if primary_face else None
        crop_params = zoomer.update(frame.shape, face_box=face_box)
        display_frame = FaceCenterZoomer.apply_zoom(frame, crop_params)

        if primary_face:
            name = primary_face["name"]
            is_known = primary_face["is_known"]

            if is_known:
                if name == last_stable_name:
                    stable_count += 1
                else:
                    last_stable_name = name
                    stable_count = 1
            else:
                last_stable_name = None
                stable_count = 0

            x1, y1, x2, y2 = transform_box_to_zoomed_view(
                primary_face["box"], crop_params, frame_w, frame_h
            )

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

            if SAVE_PHOTO_WHEN_KNOWN_STABLE and is_known and stable_count >= STABLE_FRAMES_REQUIRED:
                now = time.time()
                if now - last_capture_time_by_name[name] >= PHOTO_COOLDOWN_SECONDS:
                    clean_frame = FaceCenterZoomer.apply_zoom(frame, crop_params)
                    saved_path = save_person_photo(name, clean_frame)
                    last_capture_time_by_name[name] = now
                    print(f"Saved photo for {name}: {saved_path}")
        else:
            last_stable_name = None
            stable_count = 0

        status = f"Zoom: {zoomer.zoom:.2f}x"
        if primary_face:
            status += " | Centering face"
        else:
            status += " | Searching face"

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
            "ESC = exit",
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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
