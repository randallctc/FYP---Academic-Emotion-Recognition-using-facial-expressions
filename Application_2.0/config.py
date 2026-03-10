"""
Configuration file for Academic Emotion Recognition System
Edit these values to customize behavior
"""

# Camera Settings
CAMERA_ID = 0  # Try 0, 1, 2 if default doesn't work
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Model Settings
FRAME_SIZE = 224  # Size to resize face crops to (should match model input)
BUFFER_SIZE = 60  # Number of frames to buffer (should match model's sequence length)
MARGIN_RATIO = 0.40  # Margin around detected face (0.40 = 40% extra space)

# Face Detection Settings
FACE_DETECTION_SCALE_FACTOR = 1.2  # How much to reduce image size at each scale
FACE_DETECTION_MIN_NEIGHBORS = 5  # How many neighbors each candidate rectangle should have
FACE_DETECTION_MIN_SIZE = (60, 60)  # Minimum possible face size

# Emotion Detection Settings
EMOTION_CHANGE_THRESHOLD = 0.6  # Confidence threshold to log emotion change (0.0-1.0)
                                # Higher = fewer changes logged (only very confident predictions)
                                # Lower = more changes logged (even uncertain predictions)
                                # If your model always predicts one emotion, try lowering this
                                # Recommended: 0.5-0.7
EMOTIONS = ['Boredom', 'Confusion', 'Frustration', 'Engagement']

# Performance Settings (IMPORTANT for smooth webcam!)
PREDICTION_INTERVAL = 30  # Predict every N frames (30 = predict every 2 seconds at 15fps)
                          # Lower = more frequent predictions but laggier
                          # Higher = smoother video but less frequent updates
                          # Recommended: 15-45 frames

# Recording Settings
VIDEO_FPS = 30  # Frames per second for recorded videos
VIDEO_CODEC = 'mp4v'  # Video codec (mp4v, XVID, etc.)

# Output Settings
OUTPUT_DIR = 'annotated_videos'  # Directory for processed videos
RECORDING_PREFIX = 'emotion_recording'  # Prefix for recorded video files

# UI Settings
UI_FONT = 'Helvetica'
UI_FONT_SIZE_LARGE = 18
UI_FONT_SIZE_MEDIUM = 12
UI_FONT_SIZE_SMALL = 10

# Display Colors (BGR format for OpenCV)
COLOR_FACE_BOX = (0, 255, 0)  # Green
COLOR_RECORDING = (0, 0, 255)  # Red
COLOR_TEXT = (255, 255, 255)  # White
COLOR_OVERLAY = (0, 0, 0)  # Black

# Display Settings
DISPLAY_WIDTH = 960  # Width for display (lower = faster)
DISPLAY_HEIGHT = 540  # Height for display
UPDATE_INTERVAL_MS = 10  # GUI update interval in milliseconds
