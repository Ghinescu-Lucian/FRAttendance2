import os
import time
from collections import defaultdict

import cv2
from simple_facerec import SimpleFacerec


# ============================================================
# Configuration
# ============================================================
CAMERA_INDEX = 1                  # Change to 0 if your webcam is the default camera
ENCODINGS_DIR = "images/"
CAPTURE_DIR = "captures"
WINDOW_NAME = "Human Face Auto Zoom Recognition"

# Try to read a higher resolution frame. This helps real face recognition more than digital zoom.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Digital zoom/display behavior.
MIN_ZOOM = 1.0
MAX_ZOOM = 3.5
FACE_TARGET_HEIGHT = 0.48         # face should occupy about 48% of visible frame height
HEAD_TARGET_HEIGHT = 0.42         # used when body is detected but face is not yet detected
CENTER_SMOOTHING = 0.25
ZOOM_SMOOTHING = 0.18

# Recognition/lock behavior.
UNKNOWN_LABEL = "Unknown"
STABLE_FRAMES_REQUIRED = 5        # lock only after same known label appears N frames
PHOTO_COOLDOWN_SECONDS = 8
LOCK_AFTER_STABLE_RECOGNITION = True

# Motion detection.
MOTION_MIN_AREA = 900
MOTION_DILATE_ITERATIONS = 2

# Person detection. YOLO is recommended. HOG is only a fallback and is weaker.
USE_YOLO_PERSON_DETECTOR = True
YOLO_MODEL_PATH = "yolov8n.pt"    # pip install ultralytics; first run may download this model
YOLO_PERSON_CONFIDENCE = 0.45

os.makedirs(CAPTURE_DIR, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================
def clamp(value, low, high):
    return max(low, min(high, value))


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def is_box_center_inside(inner_box, outer_box):
    cx, cy = box_center(inner_box)
    x1, y1, x2, y2 = outer_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def expand_box(box, scale, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx, cy = box_center(box)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = int(clamp(cx - w / 2.0, 0, frame_w - 1))
    ny1 = int(clamp(cy - h / 2.0, 0, frame_h - 1))
    nx2 = int(clamp(cx + w / 2.0, 0, frame_w - 1))
    ny2 = int(clamp(cy + h / 2.0, 0, frame_h - 1))
    return nx1, ny1, nx2, ny2


def estimate_head_box_from_person(person_box):
    """
    When we see a person/body but no face yet, aim at the upper part of the body.
    This avoids zooming to moving hands and gives the face detector a better area to work with.
    """
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1

    head_x1 = int(x1 + 0.23 * w)
    head_x2 = int(x2 - 0.23 * w)
    head_y1 = int(y1)
    head_y2 = int(y1 + 0.38 * h)
    return head_x1, head_y1, head_x2, head_y2


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

    zx1 = clamp(zx1, 0, output_w - 1)
    zy1 = clamp(zy1, 0, output_h - 1)
    zx2 = clamp(zx2, 0, output_w - 1)
    zy2 = clamp(zy2, 0, output_h - 1)
    return zx1, zy1, zx2, zy2


def save_person_photo(name, zoomed_clean_frame):
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}.jpg")
    cv2.imwrite(path, zoomed_clean_frame)
    return path


# ============================================================
# Auto zoom controller
# ============================================================
class AutoZoomController:
    def __init__(self):
        self.center_x = None
        self.center_y = None
        self.zoom = 1.0
        self.frozen = False
        self.locked_name = None

    def _clamp_center(self, cx, cy, frame_w, frame_h, zoom):
        crop_w = frame_w / zoom
        crop_h = frame_h / zoom
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0
        cx = clamp(cx, half_w, frame_w - half_w)
        cy = clamp(cy, half_h, frame_h - half_h)
        return cx, cy

    def _calculate_target(self, frame_shape, target_box=None, target_type="face"):
        frame_h, frame_w = frame_shape[:2]

        if target_box is None:
            return frame_w / 2.0, frame_h / 2.0, MIN_ZOOM

        x1, y1, x2, y2 = target_box
        box_h = max(1, y2 - y1)
        target_cx, target_cy = box_center(target_box)

        if target_type == "face":
            target_zoom = (FACE_TARGET_HEIGHT * frame_h) / box_h
        else:
            target_zoom = (HEAD_TARGET_HEIGHT * frame_h) / box_h

        target_zoom = clamp(target_zoom, MIN_ZOOM, MAX_ZOOM)
        return target_cx, target_cy, target_zoom

    def freeze_on_target(self, frame_shape, target_box, target_type="face", name=None):
        frame_h, frame_w = frame_shape[:2]
        target_cx, target_cy, target_zoom = self._calculate_target(
            frame_shape, target_box, target_type
        )
        self.center_x, self.center_y = self._clamp_center(
            target_cx, target_cy, frame_w, frame_h, target_zoom
        )
        self.zoom = target_zoom
        self.frozen = True
        self.locked_name = name

    def unfreeze(self):
        self.frozen = False
        self.locked_name = None

    def update(self, frame_shape, target_box=None, target_type="face"):
        frame_h, frame_w = frame_shape[:2]

        if self.center_x is None or self.center_y is None:
            self.center_x = frame_w / 2.0
            self.center_y = frame_h / 2.0

        if self.frozen:
            self.center_x, self.center_y = self._clamp_center(
                self.center_x, self.center_y, frame_w, frame_h, self.zoom
            )
            return self.get_crop_params(frame_w, frame_h)

        target_cx, target_cy, target_zoom = self._calculate_target(
            frame_shape, target_box, target_type
        )

        self.center_x = (1.0 - CENTER_SMOOTHING) * self.center_x + CENTER_SMOOTHING * target_cx
        self.center_y = (1.0 - CENTER_SMOOTHING) * self.center_y + CENTER_SMOOTHING * target_cy
        self.zoom = (1.0 - ZOOM_SMOOTHING) * self.zoom + ZOOM_SMOOTHING * target_zoom
        self.zoom = clamp(self.zoom, MIN_ZOOM, MAX_ZOOM)

        self.center_x, self.center_y = self._clamp_center(
            self.center_x, self.center_y, frame_w, frame_h, self.zoom
        )
        return self.get_crop_params(frame_w, frame_h)

    def get_crop_params(self, frame_w, frame_h):
        crop_w = int(frame_w / self.zoom)
        crop_h = int(frame_h / self.zoom)
        left = int(round(self.center_x - crop_w / 2.0))
        top = int(round(self.center_y - crop_h / 2.0))
        left = clamp(left, 0, frame_w - crop_w)
        top = clamp(top, 0, frame_h - crop_h)
        return left, top, crop_w, crop_h, self.zoom

    @staticmethod
    def apply_zoom(frame, crop_params):
        frame_h, frame_w = frame.shape[:2]
        left, top, crop_w, crop_h, _ = crop_params
        crop = frame[top:top + crop_h, left:left + crop_w]
        return cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)


# ============================================================
# Detection logic
# ============================================================
def load_yolo_person_model():
    if not USE_YOLO_PERSON_DETECTOR:
        return None

    try:
        from ultralytics import YOLO
        return YOLO(YOLO_MODEL_PATH)
    except Exception as exc:
        print("YOLO person detector unavailable. Falling back to OpenCV HOG detector.")
        print("To use YOLO: pip install ultralytics")
        print(f"YOLO error: {exc}")
        return None


def create_hog_person_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_person_boxes(frame, yolo_model=None, hog=None):
    frame_h, frame_w = frame.shape[:2]
    persons = []

    if yolo_model is not None:
        results = yolo_model(frame, verbose=False, classes=[0], conf=YOLO_PERSON_CONFIDENCE)
        for result in results:
            if result.boxes is None:
                continue
            for b in result.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int).tolist()
                x1 = clamp(x1, 0, frame_w - 1)
                y1 = clamp(y1, 0, frame_h - 1)
                x2 = clamp(x2, 0, frame_w - 1)
                y2 = clamp(y2, 0, frame_h - 1)
                if box_area((x1, y1, x2, y2)) > 0:
                    persons.append((x1, y1, x2, y2))
        return persons

    if hog is not None:
        rects, _ = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(16, 16),
            scale=1.05,
        )
        for x, y, w, h in rects:
            persons.append((x, y, x + w, y + h))

    return persons


def detect_motion_boxes(frame, background_subtractor):
    fgmask = background_subtractor.apply(frame)

    # Remove shadows from MOG2: shadows are usually gray around value 127.
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    thresh = cv2.medianBlur(thresh, 5)
    thresh = cv2.dilate(thresh, None, iterations=MOTION_DILATE_ITERATIONS)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MOTION_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))

    return boxes


def build_face_infos(face_locations, face_names):
    faces = []
    for face_loc, name in zip(face_locations, face_names):
        box = normalize_face_location(face_loc)
        faces.append({
            "box": box,
            "name": name,
            "is_known": bool(name) and name != UNKNOWN_LABEL,
            "area": box_area(box),
        })
    return faces


def choose_moving_person(person_boxes, motion_boxes):
    """
    Select the person most related to the movement. Hands can move, but the target remains
    the person box that contains/overlaps that hand movement.
    """
    if not person_boxes:
        return None

    if not motion_boxes:
        # No motion, but there is a person. Prefer the largest person.
        return max(person_boxes, key=box_area)

    scored = []
    for person in person_boxes:
        score = 0
        for motion in motion_boxes:
            score += intersection_area(person, motion)
        scored.append((score, box_area(person), person))

    # First priority: moving person. Second priority: largest person.
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, _, best_person = scored[0]

    if best_score <= 0:
        return max(person_boxes, key=box_area)
    return best_person


def choose_face_for_person(faces, person_box):
    if not faces:
        return None

    if person_box is not None:
        inside_faces = [f for f in faces if is_box_center_inside(f["box"], person_box)]
        if inside_faces:
            known_inside = [f for f in inside_faces if f["is_known"]]
            if known_inside:
                return max(known_inside, key=lambda f: f["area"])
            return max(inside_faces, key=lambda f: f["area"])

    known_faces = [f for f in faces if f["is_known"]]
    if known_faces:
        return max(known_faces, key=lambda f: f["area"])
    return max(faces, key=lambda f: f["area"])


# ============================================================
# Main loop
# ============================================================
def main():
    sfr = SimpleFacerec()
    sfr.load_encoding_images(ENCODINGS_DIR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    yolo_model = load_yolo_person_model()
    hog = create_hog_person_detector() if yolo_model is None else None
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=32,
        detectShadows=True,
    )

    zoomer = AutoZoomController()
    last_capture_time_by_name = defaultdict(lambda: 0.0)
    stable_name = None
    stable_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        frame_h, frame_w = frame.shape[:2]

        # 1) Detect motion, but DO NOT zoom directly to motion.
        motion_boxes = detect_motion_boxes(frame, background_subtractor)

        # 2) Detect humans/persons.
        person_boxes = detect_person_boxes(frame, yolo_model=yolo_model, hog=hog)
        selected_person = choose_moving_person(person_boxes, motion_boxes)

        # 3) Detect/recognize faces.
        face_locations, face_names = sfr.detect_known_faces(frame)
        faces = build_face_infos(face_locations, face_names)
        selected_face = choose_face_for_person(faces, selected_person)

        target_box = None
        target_type = "face"

        if zoomer.frozen:
            crop_params = zoomer.update(frame.shape)
        else:
            if selected_face is not None:
                # Best case: we have an actual face. Zoom on the face.
                target_box = expand_box(selected_face["box"], 1.35, frame_w, frame_h)
                target_type = "face"
            elif selected_person is not None:
                # Fallback: person found but face not found yet. Aim at upper body/head zone.
                target_box = estimate_head_box_from_person(selected_person)
                target_type = "head_estimate"
            else:
                # No person: do not chase hands/objects. Stay wide/idle.
                target_box = None
                target_type = "idle"

            crop_params = zoomer.update(frame.shape, target_box=target_box, target_type=target_type)

        # Recognition stability and locking.
        if selected_face is not None and selected_face["is_known"]:
            current_name = selected_face["name"]
            if current_name == stable_name:
                stable_count += 1
            else:
                stable_name = current_name
                stable_count = 1

            if stable_count >= STABLE_FRAMES_REQUIRED and not zoomer.frozen:
                if LOCK_AFTER_STABLE_RECOGNITION:
                    zoomer.freeze_on_target(frame.shape, target_box, target_type="face", name=current_name)
                    crop_params = zoomer.get_crop_params(frame_w, frame_h)

                now = time.time()
                if now - last_capture_time_by_name[current_name] >= PHOTO_COOLDOWN_SECONDS:
                    zoomed_clean = AutoZoomController.apply_zoom(frame, crop_params)
                    saved_path = save_person_photo(current_name, zoomed_clean)
                    last_capture_time_by_name[current_name] = now
                    print(f"Saved photo for {current_name}: {saved_path}")
        else:
            stable_name = None
            stable_count = 0

        display_frame = AutoZoomController.apply_zoom(frame, crop_params)

        # Draw selected person/persons.
        for person in person_boxes:
            x1, y1, x2, y2 = transform_box_to_zoomed_view(person, crop_params, frame_w, frame_h)
            color = (255, 180, 0) if person == selected_person else (120, 120, 120)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display_frame,
                "person",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Draw faces.
        for face in faces:
            x1, y1, x2, y2 = transform_box_to_zoomed_view(face["box"], crop_params, frame_w, frame_h)
            color = (0, 200, 0) if face["is_known"] else (0, 0, 220)
            label = face["name"]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                display_frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                color,
                2,
            )

        # Draw target box, useful for debugging.
        if target_box is not None and not zoomer.frozen:
            x1, y1, x2, y2 = transform_box_to_zoomed_view(target_box, crop_params, frame_w, frame_h)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        status = f"Zoom: {zoomer.zoom:.2f}x"
        if zoomer.frozen:
            status += f" | LOCKED on {zoomer.locked_name or 'recognized person'}"
        elif selected_face is not None:
            status += " | Zooming on face"
        elif selected_person is not None:
            status += " | Person found, aiming at head"
        elif motion_boxes:
            status += " | Motion ignored: no person found"
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
        cv2.putText(
            display_frame,
            "ESC = exit | R = reset lock",
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
        elif key in (ord("r"), ord("R")):
            zoomer.unfreeze()
            stable_name = None
            stable_count = 0
            print("Camera lock reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
