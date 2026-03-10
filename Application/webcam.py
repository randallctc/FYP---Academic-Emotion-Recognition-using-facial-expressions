import cv2
import mediapipe as mp
import numpy as np
from collections import deque

CAMERA_ID = 0
FRAME_SIZE = 224
BUFFER_SIZE = 60

frame_buffer = deque(maxlen=BUFFER_SIZE)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)
cap = cv2.VideoCapture(CAMERA_ID)
print("Press ESC to exit")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    if results.detections:
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = x1 + bw
        y2 = y1 + bh
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
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
    cv2.imshow("Webcam - Face Buffer", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
