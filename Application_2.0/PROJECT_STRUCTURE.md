# Project Structure

```
emotion_recognition_project/
├── emotion_recognition_app.py      # Main application (command line)
├── emotion_recognition_gui.py      # GUI application (Tkinter)
├── analyze_annotations.py          # Annotation analysis tool
├── config.py                       # Configuration settings
├── test_setup.py                   # Setup verification script
├── requirements.txt                # Python dependencies
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
├── PROJECT_STRUCTURE.md            # This file
└── Multi_best.h5                   # Your model (place here)

# After running:
├── emotion_recording_*.mp4         # Recorded videos
├── emotion_recording_*_annotations.json  # Annotations
└── annotated_videos/               # Processed videos folder
    ├── [video]_annotated.mp4
    └── [video]_annotations.json
```

## File Descriptions

### Core Application Files

**emotion_recognition_app.py**
- Main command-line application
- Real-time emotion recognition
- Video recording with annotations
- Batch video processing
- Used as: `python emotion_recognition_app.py [video_file]`

**emotion_recognition_gui.py**
- User-friendly GUI application
- All features of CLI version
- Visual feedback and controls
- Progress indicators
- Used as: `python emotion_recognition_gui.py`

**config.py**
- Centralized configuration
- Camera settings
- Model parameters
- Detection thresholds
- UI preferences
- Edit this to customize behavior

### Support Files

**analyze_annotations.py**
- Analyze annotation JSON files
- Generate statistics and insights
- Export to CSV format
- Usage: `python analyze_annotations.py recording_annotations.json`

**test_setup.py**
- Verify installation
- Test dependencies
- Check camera access
- Validate face detection
- Run before first use: `python test_setup.py`

**requirements.txt**
- Python package dependencies
- Install with: `pip install -r requirements.txt`
- Contains: opencv-python, tensorflow, numpy, pillow

### Documentation

**README.md**
- Complete documentation
- Features overview
- Installation instructions
- Usage examples
- Configuration guide
- Troubleshooting

**QUICKSTART.md**
- Quick start guide
- Step-by-step setup
- Common tasks
- Best practices
- Example workflows

**PROJECT_STRUCTURE.md**
- This file
- Project organization
- File descriptions
- Usage patterns

## Key Components

### AcademicEmotionRecognizer Class
Main class handling emotion recognition:
- `__init__()`: Initialize model and settings
- `predict_emotion()`: Run model inference
- `get_dominant_emotion()`: Extract primary emotion
- `detect_emotion_change()`: Track emotion transitions
- `run_live_capture()`: Webcam capture loop
- `start_recording()`: Begin video recording
- `stop_recording()`: End recording and save

### process_video_file() Function
Process existing videos:
- Load video file
- Detect faces frame-by-frame
- Run emotion recognition
- Generate annotated output
- Save timeline JSON

### EmotionRecognitionGUI Class
GUI interface:
- `setup_ui()`: Create interface
- `load_model()`: Model loading
- `start_webcam()`: Begin capture
- `update_frame()`: Display loop
- `toggle_recording()`: Recording control
- `process_video()`: Batch processing

## Data Flow

```
Input (Webcam/Video)
    ↓
Face Detection (Haar Cascade)
    ↓
Face Crop & Resize (224x224)
    ↓
Frame Buffer (60 frames)
    ↓
ResNet50 Feature Extraction (60, 2048)
    ↓
GRU Model Prediction (Multi_best.h5)
    ↓
Emotion Classification
    ↓
Change Detection & Logging
    ↓
Output (Video + JSON)
```

## Configuration Options

Edit `config.py` to customize:
- Camera device ID
- Frame buffer size
- Face detection sensitivity
- Emotion change threshold
- Recording settings
- UI appearance

## Output Files

### Video Files (.mp4)
- Annotated with emotion overlays
- Green boxes around faces
- Real-time emotion scores
- Recording indicator
- Buffer status

### Annotation Files (.json)
```json
[
  {
    "timestamp": "00:15.450",
    "frame": 463,
    "emotion": "Engagement",
    "confidence": 0.85,
    "all_scores": {
      "Boredom": 0.08,
      "Confusion": 0.12,
      "Frustration": 0.15,
      "Engagement": 0.85
    }
  }
]
```

## Usage Patterns

### Pattern 1: Real-time Monitoring
```bash
python emotion_recognition_gui.py
# Load model → Start webcam → Monitor in real-time
```

### Pattern 2: Session Recording
```bash
python emotion_recognition_app.py
# Press R to start recording
# Study/present normally
# Press R to stop
# Annotations saved automatically
```

### Pattern 3: Batch Processing
```bash
python emotion_recognition_app.py lecture1.mp4
python emotion_recognition_app.py lecture2.mp4
# Process multiple videos
# Find results in annotated_videos/
```

### Pattern 4: Analysis
```bash
python analyze_annotations.py recording_annotations.json
# View statistics
# Export to CSV with --csv flag
```

## Dependencies

### Required
- Python 3.7+
- OpenCV (cv2)
- TensorFlow
- NumPy
- Pillow (PIL)
- Tkinter (usually included with Python)

### Optional
- CUDA (for GPU acceleration)
- cuDNN (for faster inference)

## Model Requirements

Your `Multi_best.h5` model must:
- Accept input: (batch, 60, 2048) - ResNet50 features
- Output: (batch, 4) probabilities
- Emotions order: [Boredom, Confusion, Frustration, Engagement]

**Note:** The system automatically extracts ResNet50 features from raw frames before feeding them to your GRU model.

## Best Practices

1. **Testing**: Run `test_setup.py` first
2. **Lighting**: Ensure good, even lighting
3. **Position**: Face camera at eye level
4. **Buffer**: Wait for full buffer (60 frames)
5. **Recording**: Save recordings in short sessions
6. **Analysis**: Review JSON files for insights
7. **Backup**: Keep original videos separately

## Troubleshooting

**Import errors**: Install requirements
**Camera issues**: Check CAMERA_ID in config
**Model errors**: Verify model format and path
**Low accuracy**: Check lighting and face visibility
**Performance**: Lower resolution in config

## Future Enhancements

Potential additions:
- Multi-face tracking
- Real-time graphs
- Emotion heatmaps
- Custom model training
- Cloud integration
- Mobile version
- Database storage
- Team collaboration

## License

Educational and research use.

---
For detailed usage instructions, see README.md
For quick start, see QUICKSTART.md
