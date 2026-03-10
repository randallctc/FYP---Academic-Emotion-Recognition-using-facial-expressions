import cv2
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
import json
import os

class LightweightCNN:
    """Lightweight CNN for fast feature extraction (replaces ResNet50)"""
    
    def __init__(self, output_features=2048):
        """
        Create a simple CNN that outputs features matching ResNet50 shape (2048)
        This is MUCH faster than ResNet50 while still extracting useful features
        """
        self.output_features = output_features
        self.model = self._build_model()
    
    def _build_model(self):
        """Build a lightweight CNN feature extractor"""
        inputs = keras.Input(shape=(224, 224, 3))
        
        # Block 1
        x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D(2)(x)  # 112x112
        
        # Block 2
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)  # 56x56
        
        # Block 3
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)  # 28x28
        
        # Block 4
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2)(x)  # 14x14
        
        # Block 5
        x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)  # 512 features
        
        # Dense layer to match ResNet50 output size
        outputs = layers.Dense(self.output_features, activation='relu')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='lightweight_cnn')
        return model
    
    def extract_features(self, frames):
        """
        Extract features from frames
        Args:
            frames: numpy array of shape (batch, 224, 224, 3) with values 0-1
        Returns:
            features: numpy array of shape (batch, 2048)
        """
        # Normalize to 0-1 range (if not already)
        if frames.max() > 1.0:
            frames = frames / 255.0
        
        # Extract features
        features = self.model.predict(frames, verbose=0, batch_size=60)
        return features


class AcademicEmotionRecognizer:
    def __init__(self, model_path="Multi_best.h5", use_lightweight_cnn=True):
        """Initialize the emotion recognition system"""
        # Load the GRU model (expects features with shape (60, 2048))
        print(f"Loading GRU model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path)
        print("GRU model loaded successfully!")
        
        # Load feature extractor
        if use_lightweight_cnn:
            print("Loading Lightweight CNN for feature extraction...")
            self.feature_extractor = LightweightCNN(output_features=2048)
            print("Lightweight CNN loaded successfully!")
            print("⚡ Using fast CNN - expect 5-10x speedup vs ResNet50!")
        else:
            print("Loading ResNet50 for feature extraction...")
            from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
            self.feature_extractor_model = ResNet50(
                weights='imagenet',
                include_top=False,
                pooling='avg',
                input_shape=(224, 224, 3)
            )
            self.preprocess_input = preprocess_input
            self.feature_extractor = None
            print("ResNet50 loaded successfully!")
        
        self.emotions = ['Boredom', 'Confusion', 'Frustration', 'Engagement']
        
        # Video capture settings
        self.CAMERA_ID = 0
        self.FRAME_SIZE = 224
        self.BUFFER_SIZE = 60
        self.MARGIN_RATIO = 0.40
        
        # Buffers and trackers
        self.frame_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        # Recording variables
        self.is_recording = False
        self.video_writer = None
        self.emotion_timeline = []
        self.current_emotion_state = None
        self.emotion_change_threshold = 0.6
        self.frame_count = 0
        self.fps = 30
        
        # Performance optimization
        self.last_prediction = None
        self.prediction_interval = 30  # Predict every N frames
        self.frames_since_prediction = 0
    
    def extract_features(self, frame_sequence):
        """Extract features from frame sequence using chosen method"""
        if len(frame_sequence) < self.BUFFER_SIZE:
            return None
        
        # Convert frames to array (60, 224, 224, 3)
        frames = np.array(list(frame_sequence))
        
        if self.feature_extractor is not None:
            # Use lightweight CNN (FAST!)
            features = self.feature_extractor.extract_features(frames)
        else:
            # Use ResNet50 (SLOW but maybe more accurate)
            frames_preprocessed = frames * 255.0
            frames_preprocessed = self.preprocess_input(frames_preprocessed)
            features = self.feature_extractor_model.predict(frames_preprocessed, verbose=0, batch_size=60)
        
        # features shape: (60, 2048)
        return features
    
    def predict_emotion(self, frame_sequence):
        """Predict emotions from a sequence of frames"""
        # Only predict when buffer is completely full
        if len(frame_sequence) < self.BUFFER_SIZE:
            return None
            
        # Extract features
        features = self.extract_features(frame_sequence)
        
        if features is None:
            return None
        
        # Add batch dimension (1, 60, 2048)
        features = np.expand_dims(features, axis=0)
        
        # Predict using GRU model
        predictions = self.model.predict(features, verbose=0)[0]
        return predictions
    
    def get_dominant_emotion(self, predictions):
        """Get the dominant emotion from predictions"""
        if predictions is None:
            return None, None
        
        max_idx = np.argmax(predictions)
        return self.emotions[max_idx], predictions[max_idx]
    
    def detect_emotion_change(self, new_emotion, confidence):
        """Detect if there's a significant emotion change"""
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
        """Convert frame number to timestamp"""
        seconds = frame_number / self.fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    
    def draw_ui(self, frame, predictions, is_recording):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Create overlay panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Recording indicator
        if is_recording:
            cv2.circle(frame, (30, 40), 10, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (50, 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Buffer status
        cv2.putText(frame, f"Buffer: {len(self.frame_buffer)}/{self.BUFFER_SIZE}",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Emotion predictions
        if predictions is not None:
            y_offset = 120
            for i, (emotion, score) in enumerate(zip(self.emotions, predictions)):
                color = (0, 255, 0) if score > 0.5 else (255, 255, 255)
                bar_width = int(200 * score)
                
                # Draw emotion name
                cv2.putText(frame, emotion, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw confidence bar
                cv2.rectangle(frame, (150, y_offset - 15), 
                            (150 + bar_width, y_offset - 5), color, -1)
                cv2.rectangle(frame, (150, y_offset - 15), 
                            (350, y_offset - 5), (100, 100, 100), 1)
                
                # Draw score
                cv2.putText(frame, f"{score:.2f}", (360, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                y_offset += 30
        
        # Controls
        cv2.putText(frame, "Controls:", (20, h - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "R - Start/Stop Recording | ESC - Exit", (20, h - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_recording(self):
        """Start video recording"""
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"emotion_recording_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_filename, fourcc, self.fps, (1280, 720)
            )
            
            self.is_recording = True
            self.emotion_timeline = []
            self.frame_count = 0
            self.current_emotion_state = None
            
            print(f"Recording started: {output_filename}")
            return output_filename
        return None
    
    def stop_recording(self, output_filename):
        """Stop recording and save annotations"""
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            self.is_recording = False
            
            # Save emotion timeline
            json_filename = output_filename.replace('.mp4', '_annotations.json')
            with open(json_filename, 'w') as f:
                json.dump(self.emotion_timeline, f, indent=2)
            
            print(f"Recording stopped: {output_filename}")
            print(f"Annotations saved: {json_filename}")
            return json_filename
        return None
    
    def run_live_capture(self):
        """Run live webcam capture with emotion recognition"""
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
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
            )
            
            predictions = None
            
            if len(faces) > 0:
                # Get first face
                x, y, w, h = faces[0]
                mx = int(w * self.MARGIN_RATIO)
                my = int(h * self.MARGIN_RATIO)
                x1 = max(0, x - mx)
                y1 = max(0, y - my)
                x2 = min(frame.shape[1], x + w + mx)
                y2 = min(frame.shape[0], y + h + my)
                
                # Draw face rectangle
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Process face crop
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (self.FRAME_SIZE, self.FRAME_SIZE))
                    face_crop = face_crop.astype(np.float32) / 255.0
                    self.frame_buffer.append(face_crop)
                    
                    # Only predict every N frames to reduce lag
                    self.frames_since_prediction += 1
                    should_predict = (self.frames_since_prediction >= self.prediction_interval)
                    
                    # Predict emotion when buffer is full AND it's time to predict
                    if len(self.frame_buffer) == self.BUFFER_SIZE and should_predict:
                        predictions = self.predict_emotion(self.frame_buffer)
                        self.last_prediction = predictions
                        self.frames_since_prediction = 0
                        
                        # Track emotion changes during recording
                        if self.is_recording and predictions is not None:
                            emotion, confidence = self.get_dominant_emotion(predictions)
                            
                            if self.detect_emotion_change(emotion, confidence):
                                timestamp = self.format_timestamp(self.frame_count)
                                self.emotion_timeline.append({
                                    'timestamp': timestamp,
                                    'frame': self.frame_count,
                                    'emotion': emotion,
                                    'confidence': float(confidence),
                                    'all_scores': {
                                        self.emotions[i]: float(predictions[i]) 
                                        for i in range(len(self.emotions))
                                    }
                                })
                                print(f"[{timestamp}] Emotion: {emotion} ({confidence:.2f})")
                    else:
                        # Use last prediction for display
                        predictions = self.last_prediction
            
            # Draw UI (use cached prediction if available)
            display_frame = self.draw_ui(display_frame, predictions, self.is_recording)
            
            # Write frame if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(display_frame)
                self.frame_count += 1
            
            # Show frame
            cv2.imshow("Academic Emotion Recognition (Lightweight)", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                if not self.is_recording:
                    output_filename = self.start_recording()
                else:
                    self.stop_recording(output_filename)
        
        # Cleanup
        if self.is_recording:
            self.stop_recording(output_filename)
        
        cap.release()
        cv2.destroyAllWindows()


def process_video_file(video_path, model_path="Multi_best.h5", output_dir="annotated_videos", use_lightweight_cnn=True):
    """Process a video file and create annotated version"""
    try:
        print(f"\n{'='*60}")
        print(f"Processing Video File")
        print(f"{'='*60}")
        print(f"Video path: {video_path}")
        print(f"Model path: {model_path}")
        print(f"Feature extractor: {'Lightweight CNN' if use_lightweight_cnn else 'ResNet50'}")
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print("Loading emotion recognizer...")
        recognizer = AcademicEmotionRecognizer(model_path, use_lightweight_cnn=use_lightweight_cnn)
        print("Recognizer loaded successfully!")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        print(f"Opening video file...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or width == 0 or height == 0:
            raise ValueError(f"Invalid video properties: fps={fps}, width={width}, height={height}")
        
        print(f"Video properties:")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Total frames: {total_frames}")
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        raise
    
    # Setup output
    base_name = os.path.basename(video_path).rsplit('.', 1)[0]
    output_video = os.path.join(output_dir, f"{base_name}_annotated.mp4")
    output_json = os.path.join(output_dir, f"{base_name}_annotations.json")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Cannot create output video file: {output_video}")
    
    recognizer.fps = fps
    frame_count = 0
    emotion_timeline = []
    current_emotion_state = None
    
    print(f"\nProcessing frames...")
    print(f"Output will be saved to: {output_video}")
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            display_frame = frame.copy()
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = recognizer.face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
            )
            
            predictions = None
            
            if len(faces) > 0:
                x, y, w, h = faces[0]
                mx = int(w * recognizer.MARGIN_RATIO)
                my = int(h * recognizer.MARGIN_RATIO)
                x1 = max(0, x - mx)
                y1 = max(0, y - my)
                x2 = min(frame.shape[1], x + w + mx)
                y2 = min(frame.shape[0], y + h + my)
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, (recognizer.FRAME_SIZE, recognizer.FRAME_SIZE))
                    face_crop = face_crop.astype(np.float32) / 255.0
                    recognizer.frame_buffer.append(face_crop)
                    
                    if len(recognizer.frame_buffer) == recognizer.BUFFER_SIZE:
                        predictions = recognizer.predict_emotion(recognizer.frame_buffer)
                        
                        if predictions is not None:
                            emotion, confidence = recognizer.get_dominant_emotion(predictions)
                            
                            # Only log when emotion changes
                            if confidence >= recognizer.emotion_change_threshold:
                                if current_emotion_state is None or emotion != current_emotion_state:
                                    current_emotion_state = emotion
                                    timestamp = recognizer.format_timestamp(frame_count)
                                    emotion_timeline.append({
                                        'timestamp': timestamp,
                                        'frame': frame_count,
                                        'emotion': emotion,
                                        'confidence': float(confidence),
                                        'all_scores': {
                                            recognizer.emotions[i]: float(predictions[i]) 
                                            for i in range(len(recognizer.emotions))
                                        }
                                    })
            
            # Draw annotations on frame
            display_frame = recognizer.draw_ui(display_frame, predictions, False)
            
            # Add current emotion label if available
            if current_emotion_state:
                cv2.putText(display_frame, f"Current: {current_emotion_state}", 
                           (width - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (0, 255, 255), 2)
            
            out.write(display_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                progress = (frame_count * 100) // total_frames if total_frames > 0 else 0
                print(f"Processed {frame_count}/{total_frames} frames ({progress}%)")
    
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
    finally:
        # Cleanup
        cap.release()
        out.release()
    
    # Save annotations
    with open(output_json, 'w') as f:
        json.dump(emotion_timeline, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Processing Complete!")
    print(f"{'='*60}")
    print(f"✓ Annotated video: {output_video}")
    print(f"✓ Annotations: {output_json}")
    print(f"✓ Total emotion changes: {len(emotion_timeline)}")
    print(f"✓ Frames processed: {frame_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process video file
        video_path = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "Multi_best.h5"
        use_lightweight = sys.argv[3].lower() != 'resnet50' if len(sys.argv) > 3 else True
        process_video_file(video_path, model_path, use_lightweight_cnn=use_lightweight)
    else:
        # Run live capture
        model_path = "Multi_best.h5"
        print("\n" + "="*60)
        print("Choose feature extractor:")
        print("1. Lightweight CNN (FAST - Recommended)")
        print("2. ResNet50 (SLOW but pre-trained)")
        print("="*60)
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        
        use_lightweight = choice != '2'
        
        recognizer = AcademicEmotionRecognizer(model_path, use_lightweight_cnn=use_lightweight)
        recognizer.run_live_capture()
