import cv2
import numpy as np
from simple_facerec import SimpleFacerec


# =========================
# Configuration
# =========================
CAMERA_INDEX = 1
ENCODINGS_DIR = "images/"

WINDOW_NAME = "Zoomed Face Recognition - Enhanced"

ZOOM_FACTOR = 5.5

# Enhancement options
ENHANCE_FOR_RECOGNITION = True
SHOW_NIGHT_VIEW = True

# CLAHE improves local contrast.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Gamma < 1 brightens shadows.
LOW_LIGHT_GAMMA = 0.65

# Denoising helps low-light frames, but it costs speed.
ENABLE_DENOISE = False

UNKNOWN_LABEL = "Unknown"


def apply_gamma_correction(frame, gamma):
    """
    gamma < 1.0 brightens the image.
    gamma > 1.0 darkens the image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(frame, table)


def enhance_contrast_clahe(frame):
    """
    Enhances contrast on the luminance channel only.
    This is usually better than applying histogram equalization directly on BGR.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID_SIZE
    )

    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))

    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def enhance_for_face_recognition(frame):
    """
    Improves visibility for face detection/recognition:
    1. local contrast enhancement
    2. optional brightness/gamma boost in low light
    3. optional denoise

    Note: too much enhancement can hurt recognition, so keep this moderate.
    """
    enhanced = enhance_contrast_clahe(frame)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()

    # Brighten only when the frame is dark.
    if mean_brightness < 90:
        enhanced = apply_gamma_correction(enhanced, LOW_LIGHT_GAMMA)

    if ENABLE_DENOISE:
        enhanced = cv2.fastNlMeansDenoisingColored(
            enhanced,
            None,
            h=5,
            hColor=5,
            templateWindowSize=7,
            searchWindowSize=21
        )

    return enhanced


# Initialize face recognizer
sfr = SimpleFacerec()
sfr.load_encoding_images(ENCODINGS_DIR)

# Load Camera
cap = cv2.VideoCapture(CAMERA_INDEX)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # Calculate crop area for digital zoom
    new_width = int(width / ZOOM_FACTOR)
    new_height = int(height / ZOOM_FACTOR)

    crop_x1 = (width - new_width) // 2
    crop_y1 = (height - new_height) // 2
    crop_x2 = crop_x1 + new_width
    crop_y2 = crop_y1 + new_height

    # Crop and resize to simulate zoom
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    frame_zoomed = cv2.resize(
        cropped_frame,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    # Enhanced frame is used for recognition.
    # You can also display it as a "night view".
    if ENHANCE_FOR_RECOGNITION:
        recognition_frame = enhance_for_face_recognition(frame_zoomed)
    else:
        recognition_frame = frame_zoomed

    if SHOW_NIGHT_VIEW:
        display_frame = recognition_frame.copy()
    else:
        display_frame = frame_zoomed.copy()

    # Detect faces on the enhanced/zoomed frame
    face_locations, face_names = sfr.detect_known_faces(recognition_frame)

    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Green for known people, red for unknown
        color = (0, 255, 0) if name != UNKNOWN_LABEL else (0, 0, 255)

        cv2.putText(
            display_frame,
            name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            color,
            2
        )

        cv2.rectangle(
            display_frame,
            (x1, y1),
            (x2, y2),
            color,
            4
        )

    mode_text = "Night/contrast view ON" if SHOW_NIGHT_VIEW else "Normal view"
    cv2.putText(
        display_frame,
        mode_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        "ESC = exit | N = night view on/off | E = enhancement on/off | D = denoise on/off",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(WINDOW_NAME, display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key in (ord("n"), ord("N")):
        SHOW_NIGHT_VIEW = not SHOW_NIGHT_VIEW
        print(f"SHOW_NIGHT_VIEW = {SHOW_NIGHT_VIEW}")

    if key in (ord("e"), ord("E")):
        ENHANCE_FOR_RECOGNITION = not ENHANCE_FOR_RECOGNITION
        print(f"ENHANCE_FOR_RECOGNITION = {ENHANCE_FOR_RECOGNITION}")

    if key in (ord("d"), ord("D")):
        ENABLE_DENOISE = not ENABLE_DENOISE
        print(f"ENABLE_DENOISE = {ENABLE_DENOISE}")

cap.release()
cv2.destroyAllWindows()
