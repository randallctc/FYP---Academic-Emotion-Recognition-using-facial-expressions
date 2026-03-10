import cv2
import numpy as np
from collections import deque

CAMERA_ID = 0
FRAME_SIZE = 224
BUFFER_SIZE = 60
MARGIN_RATIO = 0.40

frame_buffer = deque(maxlen=BUFFER_SIZE)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
cap = cv2.VideoCapture(CAMERA_ID)
print("Press ESC to exit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) > 0:
        x, y, w, h = faces[0]
        mx = int(w * MARGIN_RATIO)
        my = int(h * MARGIN_RATIO)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(frame.shape[1], x + w + mx)
        y2 = min(frame.shape[0], y + h + my)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_crop = cv2.resize(face_crop, (FRAME_SIZE, FRAME_SIZE))
            face_crop = face_crop.astype(np.float32) / 255.0
            frame_buffer.append(face_crop)
        cv2.putText(
            frame,
            f"Buffer: {len(frame_buffer)}/{BUFFER_SIZE}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Webcam - Face Buffer (OpenCV)", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
