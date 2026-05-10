import cv2
from simple_facerec import SimpleFacerec

# Initialize face recognizer
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

zoom_factor = 1.0  # Initial zoom
zoom_step = 0.1
max_zoom = 10.0
min_zoom = 1.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Calculate crop area
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    
    if new_width <= 0 or new_height <= 0:
        continue  # Skip this frame if crop area becomes invalid

    x1 = (width - new_width) // 2
    y1 = (height - new_height) // 2
    x2 = x1 + new_width
    y2 = y1 + new_height

    # Crop and resize (simulate zoom)
    cropped_frame = frame[y1:y2, x1:x2]
    frame_zoomed = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LINEAR)

    # Detect faces on the zoomed frame
    face_locations, face_names = sfr.detect_known_faces(frame_zoomed)
    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame_zoomed, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame_zoomed, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Display zoom level
    cv2.putText(frame_zoomed, f"Zoom: {zoom_factor:.1f}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)

    cv2.imshow("Zoomed Frame", frame_zoomed)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('1'):  # Zoom in
        zoom_factor = min(max_zoom, round(zoom_factor + zoom_step, 1))
    elif key == ord('2'):  # Zoom out
        zoom_factor = max(min_zoom, round(zoom_factor - zoom_step, 1))

cap.release()
cv2.destroyAllWindows()
