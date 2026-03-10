# Architecture: ResNet50 + GRU Pipeline

## Overview

This system uses a **two-stage architecture** for academic emotion recognition:

1. **ResNet50** - Pre-trained feature extractor (frozen)
2. **GRU Model** - Your trained temporal model (Multi_best.h5)

## Pipeline Breakdown

### Stage 1: ResNet50 Feature Extraction

**Input:** 60 frames of 224×224 RGB images
```
Shape: (60, 224, 224, 3)
```

**Process:**
1. Each frame is preprocessed using ResNet50's preprocessing function
2. ResNet50 (with average pooling) extracts spatial features
3. Each frame → 2048-dimensional feature vector

**Output:** Sequence of feature vectors
```
Shape: (60, 2048)
```

**ResNet50 Configuration:**
- **Weights:** ImageNet pre-trained
- **Include top:** False (no classification layer)
- **Pooling:** Global average pooling
- **Output:** 2048-dimensional feature vector per frame

### Stage 2: GRU Emotion Classification

**Input:** Sequence of ResNet50 features
```
Shape: (60, 2048)
```

**Your Model (Multi_best.h5):**
- Processes temporal patterns across 60 frames
- Learns emotion dynamics over time
- Outputs probabilities for 4 academic emotions

**Output:** Emotion probabilities
```
Shape: (4,)
Values: [Boredom, Confusion, Frustration, Engagement]
Range: 0.0 to 1.0 (sigmoid activation)
```

## Why This Architecture?

### Separation of Concerns

**ResNet50 handles:**
- Spatial feature extraction
- Low-level visual patterns
- Face representation
- Robust to variations in lighting, angle, etc.

**GRU handles:**
- Temporal patterns
- Emotion dynamics over time
- Sequence modeling
- Context-aware predictions

### Benefits

1. **Transfer Learning:** ResNet50 provides rich pre-trained features
2. **Efficiency:** GRU only needs to learn temporal patterns (smaller model)
3. **Modularity:** Can swap either component independently
4. **Performance:** ResNet50 features are highly effective for visual tasks

## Implementation Details

### Feature Extraction Code

```python
def extract_resnet_features(self, frame_sequence):
    """Extract ResNet50 features from frame sequence"""
    # frames shape: (60, 224, 224, 3) with values 0-1
    frames = np.array(list(frame_sequence))
    
    # Convert to 0-255 range
    frames_preprocessed = frames * 255.0
    
    # Apply ResNet50 preprocessing (mean subtraction, etc.)
    frames_preprocessed = preprocess_input(frames_preprocessed)
    
    # Extract features
    # Output shape: (60, 2048)
    features = self.resnet_model.predict(frames_preprocessed, verbose=0)
    
    return features
```

### Prediction Code

```python
def predict_emotion(self, frame_sequence):
    """Predict emotions from frames"""
    # Extract features: (60, 224, 224, 3) → (60, 2048)
    features = self.extract_resnet_features(frame_sequence)
    
    # Add batch dimension: (60, 2048) → (1, 60, 2048)
    features = np.expand_dims(features, axis=0)
    
    # GRU prediction: (1, 60, 2048) → (1, 4)
    predictions = self.model.predict(features, verbose=0)[0]
    
    return predictions  # Shape: (4,)
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Raw Video Frame                      │
│                     (H, W, 3)                          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Face Detection & Crop                     │
│            Haar Cascade Classifier                      │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                 Resize to 224×224                       │
│              Normalize to [0, 1]                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Frame Buffer (60 frames)                   │
│                (60, 224, 224, 3)                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         STAGE 1: ResNet50 Feature Extraction            │
│  ┌───────────────────────────────────────────────┐     │
│  │ Input: (60, 224, 224, 3)                      │     │
│  │                                               │     │
│  │ Preprocessing:                                │     │
│  │  • Scale to [0, 255]                         │     │
│  │  • Apply ResNet50 preprocessing              │     │
│  │                                               │     │
│  │ ResNet50 (ImageNet weights):                 │     │
│  │  • Conv layers extract features              │     │
│  │  • Global average pooling                    │     │
│  │                                               │     │
│  │ Output: (60, 2048)                           │     │
│  └───────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│          STAGE 2: GRU Emotion Classification            │
│  ┌───────────────────────────────────────────────┐     │
│  │ Input: (1, 60, 2048)                         │     │
│  │                                               │     │
│  │ Your GRU Model (Multi_best.h5):              │     │
│  │  • GRU/LSTM layers process sequence          │     │
│  │  • Learn temporal emotion patterns           │     │
│  │  • Dense layer with sigmoid activation       │     │
│  │                                               │     │
│  │ Output: (1, 4)                               │     │
│  │  [Boredom, Confusion, Frustration,           │     │
│  │   Engagement]                                │     │
│  └───────────────────────────────────────────────┘     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Emotion Probabilities                      │
│     Boredom: 0.12    Confusion: 0.08                   │
│     Frustration: 0.15  Engagement: 0.85                │
└─────────────────────────────────────────────────────────┘
```

## Performance Considerations

### Memory Usage

- **ResNet50:** ~98 MB (frozen weights)
- **Frame buffer:** 60 × 224 × 224 × 3 × 4 bytes ≈ 36 MB
- **Features:** 60 × 2048 × 4 bytes ≈ 0.5 MB
- **Your GRU model:** Depends on architecture (typically 1-10 MB)

### Inference Speed

**Per prediction (60 frames):**
- ResNet50 extraction: ~100-300ms (CPU) or ~10-30ms (GPU)
- GRU inference: ~10-50ms
- Total: ~110-350ms per prediction

**Real-time performance:**
- New prediction every 60 frames
- At 30 fps: prediction every 2 seconds
- Sufficient for emotion tracking

### Optimization Tips

1. **Use GPU:** Install `tensorflow-gpu` for ~10x speedup
2. **Batch processing:** Process multiple videos at once
3. **Feature caching:** Save extracted features for re-analysis
4. **Model quantization:** Reduce model size if needed

## Training Your Own Model

If you want to train a new GRU model:

### Step 1: Extract Features

```python
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# Load ResNet50
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Extract features from your training videos
for video in training_videos:
    frames = load_frames(video, num_frames=60)  # (60, 224, 224, 3)
    frames = preprocess_input(frames * 255.0)
    features = resnet.predict(frames)  # (60, 2048)
    save_features(features, labels)
```

### Step 2: Train GRU

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

model = Sequential([
    GRU(128, return_sequences=True, input_shape=(60, 2048)),
    Dropout(0.3),
    GRU(64),
    Dropout(0.3),
    Dense(4, activation='sigmoid')  # 4 emotions
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Multi-label
    metrics=['accuracy']
)

model.fit(features, labels, epochs=50, batch_size=32)
model.save('Multi_best.h5')
```

## Troubleshooting

### Common Issues

**"Model input shape mismatch"**
- Ensure your model expects (batch, 60, 2048)
- Check with: `model.input_shape`

**"Slow inference"**
- Install CUDA/cuDNN for GPU support
- Reduce video resolution
- Process fewer frames (adjust BUFFER_SIZE)

**"ResNet50 download fails"**
- Pre-download weights: `wget https://...`
- Use local weights file

**"Out of memory"**
- Reduce batch size
- Process videos individually
- Use float16 instead of float32

## References

- **ResNet50 Paper:** "Deep Residual Learning for Image Recognition" (He et al., 2015)
- **GRU Paper:** "Learning Phrase Representations using RNN Encoder-Decoder" (Cho et al., 2014)
- **Keras ResNet50:** https://keras.io/api/applications/resnet/
