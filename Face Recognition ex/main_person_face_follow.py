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
WINDOW_NAME = "Person Face Follow Auto Zoom"

# Use the highest stable resolution your camera supports.
# Digital zoom cannot create new details, so camera resolution matters a lot.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

UNKNOWN_LABEL = "Unknown"

# Digital zoom parameters.
MIN_ZOOM = 1.0
MAX_ZOOM = 2.4            # Too high makes the image bigger but blurrier.
FACE_TARGET_HEIGHT = 0.36 # Face should occupy about 36% of the visible frame height.
HEAD_TARGET_HEIGHT = 0.40 # Used only when a person is found but the face is not yet found.

CENTER_SMOOTHING = 0.20   # Lower = smoother, higher = faster follow.
ZOOM_SMOOTHING = 0.14

# Recognition stability. Do not lock/save after one random label.
STABLE_FRAMES_REQUIRED = 5
PHOTO_COOLDOWN_SECONDS = 10

# Person fallback: used only when no face is visible.
# It estimates the head area from a human/person box, but it does NOT draw person rectangles.
USE_PERSON_FALLBACK_WHEN_NO_FACE = True
PERSON_FALLBACK_EVERY_N_FRAMES = 5
HOG_MAX_WIDTH = 640

# Tracker after successful recognition.
# If recognition temporarily fails while the person walks, the tracker keeps following the face/head area.
USE_TRACKER_AFTER_RECOGNITION = True
TRACKER_BOX_EXPAND = 1.8

# Display/saving.
DRAW_FACE_RECTANGLE = True
SAVE_ZOOMED_PERSON_PHOTO = True
SAVE_FACE_CROP_PHOTO = True


os.makedirs(CAPTURE_DIR, exist_ok=True)


class FaceZoomController:
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

    def update(self, frame_shape, target_box=None, target_kind="face"):
        frame_h, frame_w = frame_shape[:2]

        if self.center_x is None or self.center_y is None:
            self.center_x = frame_w / 2.0
            self.center_y = frame_h / 2.0

        if target_box is None:
            target_cx = frame_w / 2.0
            target_cy = frame_h / 2.0
            target_zoom = MIN_ZOOM
        else:
            x1, y1, x2, y2 = target_box
            box_h = max(1, y2 - y1)
            target_cx = (x1 + x2) / 2.0
            target_cy = (y1 + y2) / 2.0

            if target_kind == "head":
                target_zoom = (HEAD_TARGET_HEIGHT * frame_h) / box_h
            else:
                target_zoom = (FACE_TARGET_HEIGHT * frame_h) / box_h

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


def clip_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(frame_w - 1, x1)))
    y1 = int(max(0, min(frame_h - 1, y1)))
    x2 = int(max(0, min(frame_w - 1, x2)))
    y2 = int(max(0, min(frame_h - 1, y2)))

    if x2 <= x1:
        x2 = min(frame_w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_h - 1, y1 + 1)

    return x1, y1, x2, y2


def expand_box(box, scale, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return clip_box((cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0), frame_w, frame_h)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


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


def build_face_candidates(face_locations, face_names, frame_w, frame_h):
    candidates = []
    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0

    for face_loc, name in zip(face_locations, face_names):
        box = clip_box(normalize_face_location(face_loc), frame_w, frame_h)
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_distance = ((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5
        is_known = bool(name) and name != UNKNOWN_LABEL

        candidates.append({
            "box": box,
            "name": name,
            "is_known": is_known,
            "area": box_area(box),
            "center_distance": center_distance,
        })

    return candidates


def choose_primary_face(candidates, locked_name=None):
    if not candidates:
        return None

    if locked_name:
        same_person = [f for f in candidates if f["name"] == locked_name]
        if same_person:
            same_person.sort(key=lambda f: (-f["area"], f["center_distance"]))
            return same_person[0]

    # In search mode, choose one face only.
    # Known faces first, then larger faces, then more central faces.
    candidates = list(candidates)
    candidates.sort(key=lambda f: (not f["is_known"], -f["area"], f["center_distance"]))
    return candidates[0]


def transform_box_to_zoomed_view(box, crop_params, output_w, output_h):
    x1, y1, x2, y2 = box
    left, top, crop_w, crop_h = crop_params

    sx = output_w / crop_w
    sy = output_h / crop_h

    return clip_box(
        (
            int((x1 - left) * sx),
            int((y1 - top) * sy),
            int((x2 - left) * sx),
            int((y2 - top) * sy),
        ),
        output_w,
        output_h,
    )


def transform_box_from_zoomed_view(box, crop_params, source_w, source_h):
    x1, y1, x2, y2 = box
    left, top, crop_w, crop_h = crop_params

    sx = crop_w / source_w
    sy = crop_h / source_h

    return (
        int(left + x1 * sx),
        int(top + y1 * sy),
        int(left + x2 * sx),
        int(top + y2 * sy),
    )


def create_tracker():
    """
    Creates the best available OpenCV tracker.
    Different OpenCV builds expose trackers in different namespaces.
    """
    creators = [
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerKCF_create(),
        lambda: cv2.legacy.TrackerMOSSE_create(),
    ]

    for creator in creators:
        try:
            return creator()
        except Exception:
            continue
    return None


def init_tracker(frame, face_box):
    if not USE_TRACKER_AFTER_RECOGNITION:
        return None

    frame_h, frame_w = frame.shape[:2]
    track_box = expand_box(face_box, TRACKER_BOX_EXPAND, frame_w, frame_h)
    x1, y1, x2, y2 = track_box
    tracker = create_tracker()
    if tracker is None:
        return None

    try:
        ok = tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
        return tracker if ok is not False else None
    except Exception:
        return None


def update_tracker(tracker, frame):
    if tracker is None:
        return None

    try:
        ok, rect = tracker.update(frame)
    except Exception:
        return None

    if not ok:
        return None

    x, y, w, h = rect
    frame_h, frame_w = frame.shape[:2]
    return clip_box((x, y, x + w, y + h), frame_w, frame_h)


def create_hog_person_detector():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def detect_person_boxes_hog(frame, hog):
    """
    Fallback only: detect full human body and estimate the head area.
    This is not used for recognition. It only helps the camera aim when no face is visible yet.
    """
    frame_h, frame_w = frame.shape[:2]
    scale = 1.0
    small = frame

    if frame_w > HOG_MAX_WIDTH:
        scale = HOG_MAX_WIDTH / float(frame_w)
        small = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))

    rects, weights = hog.detectMultiScale(
        small,
        winStride=(8, 8),
        padding=(8, 8),
        scale=1.05,
    )

    boxes = []
    for rect, weight in zip(rects, weights):
        if weight < 0.25:
            continue
        x, y, w, h = rect
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + w) / scale)
        y2 = int((y + h) / scale)
        boxes.append(clip_box((x1, y1, x2, y2), frame_w, frame_h))

    return boxes


def estimate_head_box_from_person(person_box, frame_w, frame_h):
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1

    # Approximate head/face area from the upper body.
    head_w = w * 0.55
    head_h = h * 0.30
    head_cx = (x1 + x2) / 2.0
    head_y1 = y1 + h * 0.03
    head_y2 = head_y1 + head_h

    return clip_box(
        (
            head_cx - head_w / 2.0,
            head_y1,
            head_cx + head_w / 2.0,
            head_y2,
        ),
        frame_w,
        frame_h,
    )


def choose_person_head_fallback(frame, hog):
    frame_h, frame_w = frame.shape[:2]
    person_boxes = detect_person_boxes_hog(frame, hog)
    if not person_boxes:
        return None

    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0

    def score(person_box):
        x1, y1, x2, y2 = person_box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        center_distance = ((cx - frame_cx) ** 2 + (cy - frame_cy) ** 2) ** 0.5
        return (-box_area(person_box), center_distance)

    best_person = sorted(person_boxes, key=score)[0]
    return estimate_head_box_from_person(best_person, frame_w, frame_h)


def save_photos(name, original_frame, zoomed_clean_frame, original_face_box=None):
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved_paths = []

    if SAVE_ZOOMED_PERSON_PHOTO:
        person_path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}_person.jpg")
        cv2.imwrite(person_path, zoomed_clean_frame)
        saved_paths.append(person_path)

    if SAVE_FACE_CROP_PHOTO and original_face_box is not None:
        frame_h, frame_w = original_frame.shape[:2]
        face_box = expand_box(original_face_box, 1.35, frame_w, frame_h)
        x1, y1, x2, y2 = face_box
        face_crop = original_frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_path = os.path.join(CAPTURE_DIR, f"{safe_name}_{timestamp}_face.jpg")
            cv2.imwrite(face_path, face_crop)
            saved_paths.append(face_path)

    return saved_paths


def main():
    sfr = SimpleFacerec()
    sfr.load_encoding_images(ENCODINGS_DIR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}. Try CAMERA_INDEX = 0.")

    hog = create_hog_person_detector() if USE_PERSON_FALLBACK_WHEN_NO_FACE else None
    zoomer = FaceZoomController()

    locked_name = None
    stable_name = None
    stable_count = 0
    last_capture_time_by_name = defaultdict(lambda: 0.0)
    tracker = None
    frame_index = 0
    last_head_fallback_box = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame from camera.")
            break

        frame_index += 1
        frame_h, frame_w = frame.shape[:2]

        # 1) Detect/recognize faces on the full camera frame.
        full_face_locations, full_face_names = sfr.detect_known_faces(frame)
        full_faces = build_face_candidates(full_face_locations, full_face_names, frame_w, frame_h)

        target_box = None
        target_kind = "face"
        target_source = "Searching face"
        face_for_tracker = None

        # 2) If a person was already identified, follow only that person.
        if locked_name:
            locked_face = choose_primary_face(full_faces, locked_name=locked_name)
            if locked_face and locked_face["name"] == locked_name:
                target_box = locked_face["box"]
                face_for_tracker = target_box
                target_source = f"Following {locked_name} by face"
                tracker = init_tracker(frame, target_box)
            else:
                tracked_box = update_tracker(tracker, frame)
                if tracked_box is not None:
                    target_box = tracked_box
                    target_kind = "head"
                    target_source = f"Following {locked_name} by tracker"
                else:
                    # If tracker fails, fall back to searching for the best visible face.
                    search_face = choose_primary_face(full_faces)
                    if search_face:
                        target_box = search_face["box"]
                        face_for_tracker = target_box
                        target_source = "Re-searching face"
                    else:
                        target_source = f"Lost {locked_name}; searching face"
        else:
            # 3) Search mode: use the best visible face as the zoom target.
            search_face = choose_primary_face(full_faces)
            if search_face:
                target_box = search_face["box"]
                face_for_tracker = target_box
                target_source = "Centering visible face"
            elif hog is not None:
                # 4) Fallback: find a person, estimate the head area, and aim there.
                # This is only for aiming. It does not identify the person.
                if frame_index % PERSON_FALLBACK_EVERY_N_FRAMES == 0:
                    last_head_fallback_box = choose_person_head_fallback(frame, hog)
                if last_head_fallback_box is not None:
                    target_box = last_head_fallback_box
                    target_kind = "head"
                    target_source = "Aiming at estimated head area"

        # 5) Center/zoom the camera view on the face/head target.
        crop_params = zoomer.update(frame.shape, target_box=target_box, target_kind=target_kind)
        zoomed_clean = FaceZoomController.apply_zoom(frame, crop_params)
        display_frame = zoomed_clean.copy()

        # 6) Run recognition again on the zoomed frame.
        # This is the pass that decides whether we can lock/save/follow a known person.
        zoom_face_locations, zoom_face_names = sfr.detect_known_faces(zoomed_clean)
        zoom_faces = build_face_candidates(zoom_face_locations, zoom_face_names, frame_w, frame_h)
        zoom_primary = choose_primary_face(zoom_faces, locked_name=locked_name)

        recognized_face_original_box = None
        display_face = None

        if zoom_primary:
            display_face = zoom_primary
            recognized_name = zoom_primary["name"]
            recognized_known = zoom_primary["is_known"]

            recognized_face_original_box = clip_box(
                transform_box_from_zoomed_view(zoom_primary["box"], crop_params, frame_w, frame_h),
                frame_w,
                frame_h,
            )

            if recognized_known:
                if recognized_name == stable_name:
                    stable_count += 1
                else:
                    stable_name = recognized_name
                    stable_count = 1

                if stable_count >= STABLE_FRAMES_REQUIRED:
                    # Lock/follow this person after repeated successful recognition.
                    if locked_name != recognized_name:
                        print(f"Locked on person: {recognized_name}")
                    locked_name = recognized_name
                    tracker = init_tracker(frame, recognized_face_original_box)

                    now = time.time()
                    if now - last_capture_time_by_name[recognized_name] >= PHOTO_COOLDOWN_SECONDS:
                        saved_paths = save_photos(
                            recognized_name,
                            original_frame=frame,
                            zoomed_clean_frame=zoomed_clean,
                            original_face_box=recognized_face_original_box,
                        )
                        last_capture_time_by_name[recognized_name] = now
                        for path in saved_paths:
                            print(f"Saved: {path}")
            else:
                # Do not reset the lock just because one frame is unknown.
                if locked_name is None:
                    stable_name = None
                    stable_count = 0
        else:
            if locked_name is None:
                stable_name = None
                stable_count = 0

        # 7) Draw one rectangle only: the face used by the zoomed recognition pass.
        if DRAW_FACE_RECTANGLE and display_face:
            x1, y1, x2, y2 = display_face["box"]
            color = (0, 180, 0) if display_face["is_known"] else (0, 0, 200)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                display_frame,
                display_face["name"],
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                color,
                2,
            )

        status = f"Zoom: {zoomer.zoom:.2f}x | {target_source}"
        if locked_name:
            status += f" | LOCKED: {locked_name}"
        elif stable_name:
            status += f" | Candidate: {stable_name} {stable_count}/{STABLE_FRAMES_REQUIRED}"

        cv2.putText(
            display_frame,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
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
        if key == 27:  # ESC
            break
        if key in (ord("r"), ord("R")):
            locked_name = None
            stable_name = None
            stable_count = 0
            tracker = None
            print("Reset followed person.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
