import cv2
from simple_facerec import SimpleFacerec

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Cameras
cap1 = cv2.VideoCapture(0)  # Laptop camera
cap2 = cv2.VideoCapture(1)  # USB camera

# Check if cameras opened
if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Cannot open one or both cameras")
    exit()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Cannot read from one or both cameras")
        break

    # Detect Faces in frame1
    face_locations1, face_names1 = sfr.detect_known_faces(frame1)
    for face_loc, name in zip(face_locations1, face_names1):
        y1, x2, y2, x1 = face_loc
        cv2.putText(frame1, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Detect Faces in frame2
    face_locations2, face_names2 = sfr.detect_known_faces(frame2)
    for face_loc, name in zip(face_locations2, face_names2):
        y1, x2, y2, x1 = face_loc
        cv2.putText(frame2, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0, 200), 4)

    # Show both frames
    cv2.imshow("Laptop Camera", frame1)
    cv2.imshow("USB Camera", frame2)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
