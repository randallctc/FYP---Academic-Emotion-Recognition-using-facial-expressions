# Academic Emotion Recognition System

A real-time emotion recognition system that detects and tracks academic emotions (Boredom, Confusion, Frustration, and Engagement) using webcam or video files.

## Features

- **Live Webcam Recognition**: Real-time emotion detection with visual feedback
- **Video Recording**: Record sessions with automatic emotion annotation
- **Batch Video Processing**: Process existing videos and generate annotated versions
- **Emotion Timeline**: Track emotion changes with timestamps
- **Easy-to-Use GUI**: User-friendly interface built with Tkinter
- **JSON Export**: Save emotion timelines for further analysis

## Requirements

```
opencv-python>=4.8.0
tensorflow>=2.13.0
numpy>=1.24.0
pillow>=10.0.0
```

## Installation

1. Install required packages:
```bash
pip install opencv-python tensorflow numpy pillow
```

2. Ensure you have your trained model file (`Multi_best.h5`) in the same directory

## Usage

### Option 1: GUI Application (Recommended)

Run the GUI application:
```bash
python emotion_recognition_gui.py
```

**Steps:**
1. Click "Select Model File (.h5)" to load your trained model
2. Click "Start Webcam" to begin live emotion recognition
3. Click "Start Recording" to record your session with annotations
4. Click "Stop Recording" when done (annotations will be saved automatically)
5. Or use "Select Video to Process" to annotate existing videos

### Option 2: Command Line (Live Capture)

Run live webcam capture with emotion recognition:
```bash
python emotion_recognition_app.py
```

**Controls:**
- Press `R` to start/stop recording
- Press `ESC` to exit

### Option 3: Command Line (Video Processing)

Process an existing video file:
```bash
python emotion_recognition_app.py path/to/video.mp4
```

Or with custom model path:
```bash
python emotion_recognition_app.py path/to/video.mp4 path/to/model.h5
```

## Output Files

### Live Recording:
- `emotion_recording_YYYYMMDD_HHMMSS.mp4` - Recorded video with annotations
- `emotion_recording_YYYYMMDD_HHMMSS_annotations.json` - Emotion timeline

### Video Processing:
- `annotated_videos/[original_name]_annotated.mp4` - Annotated video
- `annotated_videos/[original_name]_annotations.json` - Emotion timeline

## Annotation JSON Format

```json
[
  {
    "timestamp": "00:05.123",
    "frame": 150,
    "emotion": "Engagement",
    "confidence": 0.85,
    "all_scores": {
      "Boredom": 0.12,
      "Confusion": 0.15,
      "Frustration": 0.08,
      "Engagement": 0.85
    }
  }
]
```

## Model Requirements

Your model (`Multi_best.h5`) should be a **GRU model** that:
- **Input**: ResNet50 features with shape `(batch_size, 60, 2048)`
  - 60 frames (time steps)
  - 2048 features per frame (ResNet50 output)
- **Output**: 4 values (0-1) for [Boredom, Confusion, Frustration, Engagement]
- Uses sigmoid activation for multi-label classification

**Architecture Pipeline:**
```
Raw Frames (60, 224, 224, 3)
    ↓
ResNet50 Feature Extractor (automatically applied)
    ↓
Features (60, 2048)
    ↓
Your GRU Model (Multi_best.h5)
    ↓
Predictions [Boredom, Confusion, Frustration, Engagement]
```

The system automatically handles ResNet50 feature extraction, so you only need to provide your trained GRU model.

## Configuration

Edit the constants in `emotion_recognition_app.py`:

```python
CAMERA_ID = 0              # Webcam device ID
FRAME_SIZE = 224           # Input frame size for model
BUFFER_SIZE = 60           # Number of frames to buffer
MARGIN_RATIO = 0.40        # Face crop margin (40%)
emotion_change_threshold = 0.6  # Confidence threshold for emotion changes
```

## Troubleshooting

**Camera not working:**
- Check `CAMERA_ID` value (try 0, 1, 2)
- Ensure no other application is using the camera
- Check camera permissions

**Model loading error:**
- Verify model file path
- Ensure TensorFlow version compatibility
- Check model input/output shapes

**Face not detected:**
- Ensure good lighting
- Face the camera directly
- Adjust `MARGIN_RATIO` if crops are too tight/loose

**Low accuracy:**
- Ensure buffer is full (60 frames)
- Check lighting conditions
- Verify model is trained on similar data

## Architecture

```
┌─────────────────────────────────────────┐
│         Webcam / Video Input            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Face Detection (Haar Cascade)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Face Crop & Preprocessing (224x224)  │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   Frame Buffer (60 frames rolling)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   ResNet50 Feature Extraction           │
│   (60, 224, 224, 3) → (60, 2048)        │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│   GRU Emotion Model (Multi_best.h5)     │
│   Outputs: [Boredom, Confusion,         │
│            Frustration, Engagement]      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│    Emotion Change Detection & Logging   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Annotated Video + JSON Timeline Output │
└─────────────────────────────────────────┘
```

## Future Enhancements

- [ ] Multi-face tracking
- [ ] Real-time emotion graphs
- [ ] Export to other formats (CSV, Excel)
- [ ] Emotion heatmap visualization
- [ ] Custom model training interface
- [ ] Cloud storage integration
- [ ] Mobile app version

## License

This project is provided as-is for educational and research purposes.

## Support

For issues or questions, please check:
1. Model compatibility with input/output shapes
2. Camera permissions and availability
3. TensorFlow/OpenCV installation
