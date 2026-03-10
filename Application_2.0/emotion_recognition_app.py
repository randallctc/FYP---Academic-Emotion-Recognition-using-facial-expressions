import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from datetime import datetime
import json
import os

# Fix for loading older Keras models with time_major parameter
import keras
from keras.layers import GRU

_original_gru_init = GRU.__init__

def _patched_gru_init(self, *args, **kwargs):
    kwargs.pop('time_major', None)
    _original_gru_init(self, *args, **kwargs)

GRU.__init__ = _patched_gru_init

# Emotion order must match training
EMOTION_ORDER = ['Boredom', 'Engagement', 'Confusion', 'Frustration']

class AcademicEmotionRecognizer:
    def __init__(self, model_path="final_model_v2.h5", threshold_path="emotion_thresholds.json"):
        print("Loading GRU model from: {}".format(model_path))
        self.model = tf.keras.models.load_model(model_path, compile=False)
        print("GRU model loaded successfully!")

        print("Loading ResNet50 for feature extraction...")
        self.resnet_model = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        self.resnet_model.trainable = False
        print("ResNet50 loaded successfully!")

        # Load per-emotion thresholds
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                self.thresholds = json.load(f)
            print("Thresholds loaded: {}".format(self.thresholds))
        else:
            print("WARNING: {} not found, using default threshold 0.5".format(threshold_path))
            self.thresholds = {e: 0.5 for e in EMOTION_ORDER}

        self.emotions = EMOTION_ORDER

        # Video / capture settings
        self.CAMERA_ID = 0
        self.FRAME_SIZE = 224
        self.BUFFER_SIZE = 60
        self.MARGIN_RATIO = 0.40

        self.frame_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.is_recording = False
        self.video_writer = None
        self.emotion_timeline = []
        self.current_emotion_state = None
        self.emotion_change_threshold = 0.6
        self.frame_count = 0
        self.fps = 30

        self.last_prediction = None
        self.prediction_interval = 30
        self.frames_since_prediction = 0

    def extract_resnet_features(self, frame_sequence):
        """Extract ResNet50 features from a list/deque of 60 frames (float32, 0-1)."""
        if len(frame_sequence) < self.BUFFER_SIZE:
            return None

        frames = np.array(list(frame_sequence))          # (60, 224, 224, 3)
        frames_preprocessed = frames * 255.0             # back to 0-255
        frames_preprocessed = preprocess_input(frames_preprocessed)  # caffe normalisation
        features = self.resnet_model.predict(frames_preprocessed, verbose=0, batch_size=60)
        return features  # (60, 2048)

    def predict_emotion(self, frame_sequence):
        """
        Returns a dict {emotion: score} or None if buffer not full.
        Scores are raw sigmoid outputs in [0, 1].
        """
        if len(frame_sequence) < self.BUFFER_SIZE:
            return None

        features = self.extract_resnet_features(frame_sequence)
        if features is None:
            return None

        features = np.expand_dims(features, axis=0)  # (1, 60, 2048)
        raw = self.model.predict(features, verbose=0)

        # raw is a dict {emotion_name: array([[score]])} for multi-output models
        predictions = {}
        if isinstance(raw, dict):
            for emotion in self.emotions:
                predictions[emotion] = float(np.squeeze(raw[emotion]))
        else:
            # Fallback: raw numpy array in EMOTION_ORDER
            raw = np.squeeze(raw)
            for i, emotion in enumerate(self.emotions):
                predictions[emotion] = float(raw[i])

        return predictions

    def get_dominant_emotion(self, predictions):
        """
        Returns (emotion_name, score) for the highest-scoring emotion
        that clears its per-emotion threshold.
        Returns ('Neutral', 0.0) if nothing clears any threshold.
        """
        if predictions is None:
            return 'Neutral', 0.0

        # Keep only emotions above their threshold
        above = {e: s for e, s in predictions.items()
                 if s >= self.thresholds.get(e, 0.5)}

        if above:
            dominant = max(above, key=above.get)
            return dominant, predictions[dominant]
        else:
            return 'Neutral', 0.0

    def get_model_size_mb(self):
        try:
            import sys
            resnet_size = sum(sys.getsizeof(v) for v in self.resnet_model.get_weights()) / (1024 * 1024)
            gru_size    = sum(sys.getsizeof(v) for v in self.model.get_weights()) / (1024 * 1024)
            return resnet_size + gru_size
        except Exception:
            return 100

    def detect_emotion_change(self, new_emotion, confidence):
        if confidence < self.emotion_change_threshold:
            return False
        if self.current_emotion_state is None:
            self.current_emotion_state = new_emotion
            return True
        if new_emotion != self.current_emotion_state:
            self.current_emotion_state = new_emotion
            return True
        return False

    def format_timestamp(self, frame_number):
        seconds      = frame_number / self.fps
        minutes      = int(seconds // 60)
        secs         = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return "{:02d}:{:02d}.{:03d}".format(minutes, secs, milliseconds)

    def draw_ui(self, frame, predictions, is_recording):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if is_recording:
            cv2.circle(frame, (30, 40), 10, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (50, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Buffer: {}/{}".format(len(self.frame_buffer), self.BUFFER_SIZE),
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if predictions is not None:
            y_offset = 120
            for emotion in self.emotions:
                score = predictions.get(emotion, 0.0)
                color = (0, 255, 0) if score > self.thresholds.get(emotion, 0.5) else (255, 255, 255)
                bar_width = int(200 * score)
                cv2.putText(frame, emotion, (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (150, y_offset - 15),
                              (150 + bar_width, y_offset - 5), color, -1)
                cv2.rectangle(frame, (150, y_offset - 15),
                              (350, y_offset - 5), (100, 100, 100), 1)
                cv2.putText(frame, "{:.2f}".format(score), (360, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                y_offset += 30

        cv2.putText(frame, "R - Start/Stop Recording | ESC - Exit",
                    (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def run_live_capture(self):
        cap = cv2.VideoCapture(self.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        output_filename = None

        print("\n=== Academic Emotion Recognition ===")
        print("Press 'R' to start/stop recording")
        print("Press 'ESC' to exit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            display_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
            )

            predictions = None

            if len(faces) > 0:
                x, y, w, h = faces[0]
                mx = int(w * self.MARGIN_RATIO)
                my = int(h * self.MARGIN_RATIO)
                x1 = max(0, x - mx)
                y1 = max(0, y - my)
                x2 = min(frame.shape[1], x + w + mx)
                y2 = min(frame.shape[0], y + h + my)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (self.FRAME_SIZE, self.FRAME_SIZE))
                    face_crop = face_crop.astype(np.float32) / 255.0
                    self.frame_buffer.append(face_crop)

                    self.frames_since_prediction += 1
                    if (len(self.frame_buffer) == self.BUFFER_SIZE and
                            self.frames_since_prediction >= self.prediction_interval):
                        predictions = self.predict_emotion(self.frame_buffer)
                        self.last_prediction = predictions
                        self.frames_since_prediction = 0

                        if self.is_recording and predictions is not None:
                            emotion, confidence = self.get_dominant_emotion(predictions)
                            if self.detect_emotion_change(emotion, confidence):
                                ts = self.format_timestamp(self.frame_count)
                                self.emotion_timeline.append({
                                    'timestamp': ts,
                                    'frame': self.frame_count,
                                    'emotion': emotion,
                                    'confidence': float(confidence),
                                    'all_scores': predictions
                                })
                                print("[{}] Emotion: {} ({:.2f})".format(ts, emotion, confidence))
                    else:
                        predictions = self.last_prediction

            display_frame = self.draw_ui(display_frame, predictions, self.is_recording)

            if self.is_recording and self.video_writer:
                self.video_writer.write(display_frame)
                self.frame_count += 1

            cv2.imshow("Academic Emotion Recognition", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('r') or key == ord('R'):
                if not self.is_recording:
                    output_filename = self.start_recording()
                else:
                    self.stop_recording(output_filename)

        if self.is_recording:
            self.stop_recording(output_filename)
        cap.release()
        cv2.destroyAllWindows()

    def start_recording(self):
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = "emotion_recording_{}.mp4".format(timestamp)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_filename, fourcc, self.fps, (1280, 720))
            self.is_recording = True
            self.emotion_timeline = []
            self.frame_count = 0
            self.current_emotion_state = None
            print("Recording started: {}".format(output_filename))
            return output_filename
        return None

    def stop_recording(self, output_filename):
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.is_recording = False
            json_filename = output_filename.replace('.mp4', '_annotations.json')
            with open(json_filename, 'w') as f:
                json.dump(self.emotion_timeline, f, indent=2)
            print("Recording stopped: {}".format(output_filename))
            print("Annotations saved: {}".format(json_filename))
            return json_filename
        return None


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "final_model_v2.h5"
    recognizer = AcademicEmotionRecognizer(model_path)
    recognizer.run_live_capture()
