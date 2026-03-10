import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model, Model

# -----------------------------
# Load GRU model
# -----------------------------
MODEL_PATH = r"C:\Users\Randall Chiang\Documents\GitHub\FYP---Academic-Emotion-Recognition-using-facial-expressions\all_emotions_final.h5"
model = load_model(MODEL_PATH, compile=False)
print("GRU Model loaded successfully!")

# -----------------------------
# Load ResNet50 feature extractor
# -----------------------------
base_cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")
feature_extractor = Model(inputs=base_cnn.input, outputs=base_cnn.output)
print("ResNet50 loaded successfully!")

# -----------------------------
# Emotion labels
# -----------------------------
EMOTIONS = ["Boredom", "Engagement", "Frustration", "Confusion"]

# -----------------------------
# Face Detector (OpenCV Haar)
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Webcam setup
# -----------------------------
cap = cv2.VideoCapture(0)
sequence = []

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Draw bounding boxes
    for (x, y, w, h) in faces:

        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Resize to match training (224x224)
        face_resized = cv2.resize(face_roi, (224, 224))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # Convert to array
        face_array = np.expand_dims(face_resized.astype(np.float32), axis=0)
        face_array = preprocess_input(face_array)

        # Extract features (2048)
        features = feature_extractor.predict(face_array, verbose=0)
        sequence.append(features[0])

        break  # Only use first detected face

    # Keep only last 60 frames
    if len(sequence) > 60:
        sequence.pop(0)

    # Show buffer progress
    buffer_text = f"Buffer: {len(sequence)}/60"
    cv2.putText(frame, buffer_text,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    # When buffer full → predict
    if len(sequence) == 60:
        input_data = np.expand_dims(sequence, axis=0)
        predictions = model.predict(input_data, verbose=0)[0]

        emotion_index = np.argmax(predictions)
        emotion_text = EMOTIONS[emotion_index]

        # Display prediction
        cv2.putText(frame, f"Emotion: {emotion_text}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        print(predictions)

    cv2.imshow("DAiSEE Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
