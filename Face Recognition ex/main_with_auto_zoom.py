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

WINDOW_NAME = "Face Recognition Auto Zoom"

MIN_ZOOM = 1.0            # 1.0 = no digital zoom
MAX_ZOOM = 3.0            # Increase if you want stronger digital zoom
FACE_TARGET_HEIGHT = 0.38 # Face should occupy ~38% of the visible frame height
MOTION_TARGET_HEIGHT = 0.45

CENTER_SMOOTHING = 0.12   # Smaller = smoother/slower movement
ZOOM_SMOOTHING = 0.08
LOCK_CENTER_SMOOTHING = 0.025
LOCK_ZOOM_SMOOTHING = 0.02

STABLE_FRAMES_REQUIRED = 8       # Known face must be stable for this many frames before taking photo
LOCK_AFTER_CAPTURE_FRAMES = 90   # Keep camera steady after good classification/photo
UNKNOWN_LABEL = "Unknown"
MOTION_MIN_AREA = 1800
PHOTO_COOLDOWN_SECONDS = 8       # Prevent repeated captures of the same person too fast


os.makedirs(CAPTURE_DIR, exist_ok=True)


class AutoZoomController:
    def __init__(self):
        self.center_x = None
        self.center_y = None
        self.zoom = 1.0
        self.lock_frames_left = 0

    def _clamp_center(self, cx, cy, frame_w, frame_h, zoom):
        crop_w = frame_w / zoom
        crop_h = frame_h / zoom
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        cx = max(half_w, min(frame_w - half_w, cx))
        cy = max(half_h, min(frame_h - half_h, cy))
        return cx, cy

    def update(self, frame_shape, target_box=None, target_type=None, lock=False):
        frame_h, frame_w = frame_shape[:2]

        if self.center_x is None or self.center_y is None:
            self.center_x = frame_w / 2.0
            self.center_y = frame_h / 2.0

        if target_box is None:
            target_cx, target_cy = frame_w / 2.0, frame_h / 2.0
            target_zoom = MIN_ZOOM
            lock = False
        else:
            x1, y1, x2, y2 = target_box
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            target_cx = (x1 + x2) / 2.0
            target_cy = (y1 + y2) / 2.0

            if target_type == "face":
                target_zoom = (FACE_TARGET_HEIGHT * frame_h) / box_h
            else:
                target_zoom = (MOTION_TARGET_HEIGHT * frame_h) / box_h

            target_zoom = max(MIN_ZOOM, min(MAX_ZOOM, target_zoom))

        if lock:
            self.lock_frames_left = max(self.lock_frames_left, LOCK_AFTER_CAPTURE_FRAMES)

        locked_now = self.lock_frames_left > 0
        center_alpha = LOCK_CENTER_SMOOTHING if locked_now else CENTER_SMOOTHING
        zoom_alpha = LOCK_ZOOM_SMOOTHING if locked_now else ZOOM_SMOOTHING

        # Smoothly move center and zoom level.
        self.center_x = (1.0 - center_alpha) * self.center_x + center_alpha * target_cx
        self.center_y = (1.0 - center_alpha) * self.center_y + center_alpha * target_cy
        self.zoom = (1.0 - zoom_alpha) * self.zoom + zoom_alpha * target_zoom
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom))

        self.center_x, self.center_y = self._clamp_center(
            self.center_x, self.center_y, frame_w, frame_h, self.zoom
        )

        if self.lock_frames_left > 0:
            self.lock_frames_left -= 1

        return self.get_crop_params(frame_w, frame_h)

    def get_crop_params(self, frame_w, frame_h):
        crop_w = int(frame_w / self.zoom)
        crop_h = int(frame_h / self.zoom)

        left = int(round(self.center_x - crop_w / 2.0))
        top = int(round(self.center_y - crop_h / 2.0))

        left = max(0, min(frame_w - crop_w, left))
        top = max(0, min(frame_h - crop_h, top))

        return left, top, crop_w, crop_h, self.zoom

    @staticmethod
    def apply_zoom(frame, crop_params):
        frame_h, frame_w = frame.shape[:2]
        left, top, crop_w, crop_h, zoom = crop_params
        crop = frame[top:top + crop_h, left:left + crop_w]
        zoomed = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
        return zoomed


def normalize_face_location(face_loc):
    """
    Supports both common coordinate forms:
    - [top, right, bottom, left]
    - [top, left, bottom, right]
    Returns: x1, y1, x2, y2
    """
    y1 = int(face_loc[0])
    xa = int(face_loc[1])
    y2 = int(face_loc[2])
    xb = int(face_loc[3])

    x1 = min(xa, xb)
    x2 = max(xa, xb)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2


def transform_box_to_zoomed_view(box, crop_params, output_w, output_h):
    x1, y1, x2, y2 = box
    left, top, crop_w, crop_h, _ = crop_params

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


def detect_motion_box(frame, previous_gray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if previous_gray is None:
        return None, gray

    delta = cv2.absdiff(previous_gray, gray)
    thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, gray

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MOTION_MIN_AREA:
        return None, gray

    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h), gray


def choose_primary_face(face_locations, face_names):
    faces = []
    for face_loc, name in zip(face_locations, face_names):
        box = normalize_face_location(face_loc)
        x1, y1, x2, y2 = box
        area = max(1, x2 - x1) * max(1, y2 - y1)
        is_known = bool(name) and name != UNKNOWN_LABEL
        faces.append({
            "name": name,
            "box": box,
            "area": area,
            "is_known": is_known,
        })

    if not faces:
        return None, []

    # Priority: largest known face. If none are known, largest detected face.
    known_faces = [f for f in faces if f["is_known"]]
    if known_faces:
        primary = max(known_faces, key=lambda f: f["area"])
    else:
        primary = max(faces, key=lambda f: f["area"])

    return primary, faces


def save_person_photo(name, zoomed_clean_frame):
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}.jpg")
    cv2.imwrite(path, zoomed_clean_frame)
    return path


def main():
    # Encode faces from a folder.
    sfr = SimpleFacerec()
    sfr.load_encoding_images(ENCODINGS_DIR)

    # Load camera.
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    zoomer = AutoZoomController()
    previous_gray = None

    stable_name = None
    stable_count = 0
    last_capture_time_by_name = defaultdict(lambda: 0.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        frame_h, frame_w = frame.shape[:2]

        # Detect faces on the original full frame.
        face_locations, face_names = sfr.detect_known_faces(frame)
        primary_face, all_faces = choose_primary_face(face_locations, face_names)

        # Detect movement as fallback when no face is visible.
        motion_box, previous_gray = detect_motion_box(frame, previous_gray)

        target_box = None
        target_type = None
        lock_camera = False

        if primary_face is not None:
            target_box = primary_face["box"]
            target_type = "face"

            if primary_face["is_known"]:
                current_name = primary_face["name"]
                if current_name == stable_name:
                    stable_count += 1
                else:
                    stable_name = current_name
                    stable_count = 1

                now = time.time()
                cooldown_ok = now - last_capture_time_by_name[current_name] >= PHOTO_COOLDOWN_SECONDS

                if stable_count >= STABLE_FRAMES_REQUIRED:
                    # A known person has been classified consistently.
                    # Lock the virtual camera and take a picture once per cooldown period.
                    lock_camera = True
                    if cooldown_ok:
                        # Crop/zoom first, save clean zoomed image without rectangles/text.
                        temp_crop_params = zoomer.update(
                            frame.shape,
                            target_box=target_box,
                            target_type=target_type,
                            lock=True,
                        )
                        zoomed_clean = AutoZoomController.apply_zoom(frame, temp_crop_params)
                        saved_path = save_person_photo(current_name, zoomed_clean)
                        last_capture_time_by_name[current_name] = now
                        print(f"Saved photo for {current_name}: {saved_path}")
                else:
                    lock_camera = False
            else:
                stable_name = None
                stable_count = 0
        elif motion_box is not None:
            target_box = motion_box
            target_type = "motion"
            stable_name = None
            stable_count = 0
        else:
            stable_name = None
            stable_count = 0

        # Update auto zoom. If a known face is stable, this keeps the view steady.
        crop_params = zoomer.update(
            frame.shape,
            target_box=target_box,
            target_type=target_type,
            lock=lock_camera,
        )
        display_frame = AutoZoomController.apply_zoom(frame, crop_params)

        # Draw face boxes after mapping original-frame coordinates into the zoomed view.
        for face in all_faces:
            name = face["name"]
            x1, y1, x2, y2 = transform_box_to_zoomed_view(
                face["box"], crop_params, frame_w, frame_h
            )

            color = (0, 180, 0) if face["is_known"] else (0, 0, 200)
            cv2.putText(
                display_frame,
                name,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                color,
                2,
            )
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

        # Draw motion box only when no face is detected.
        if primary_face is None and motion_box is not None:
            x1, y1, x2, y2 = transform_box_to_zoomed_view(motion_box, crop_params, frame_w, frame_h)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 180, 180), 2)
            cv2.putText(
                display_frame,
                "Movement detected",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 180, 180),
                2,
            )

        status = f"Zoom: {zoomer.zoom:.2f}x"
        if stable_name and stable_count >= STABLE_FRAMES_REQUIRED:
            status += f" | LOCKED on {stable_name}"
        elif target_type == "face":
            status += " | Tracking face"
        elif target_type == "motion":
            status += " | Tracking movement"
        else:
            status += " | Idle"

        cv2.putText(
            display_frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
