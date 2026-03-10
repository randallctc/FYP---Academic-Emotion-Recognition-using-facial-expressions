# Model Troubleshooting Guide

## Issue: Model Always Predicts the Same Emotion

If your model keeps predicting "Confusion" with high confidence (0.80-0.90) regardless of your actual expression, here are some possible causes and solutions:

### 🔍 Diagnosis

**Symptoms:**
- JSON only has 1 entry at timestamp 00:00.000
- Same emotion predicted throughout entire video
- High confidence (>0.7) for one emotion consistently

**This means:** Your model isn't detecting emotion changes, likely due to:

---

## 🛠️ Possible Fixes

### 1. **Lower the Confidence Threshold** (Quick Fix)

Your current threshold is `0.6`. Try lowering it:

**Edit `emotion_recognition_app.py` around line 45:**
```python
self.emotion_change_threshold = 0.3  # Lower threshold
```

Or edit the threshold when it checks:
```python
# Around line 430 in process_video_file
if confidence >= 0.3:  # Instead of recognizer.emotion_change_threshold
```

**Try values:** 0.3, 0.4, 0.5

---

### 2. **Check Model Training Data**

Your model might have been:
- **Trained on different people** - Test with faces similar to training data
- **Overfitted to one emotion** - Retrain with balanced dataset
- **Not enough training data** - Add more diverse examples

---

### 3. **Verify Model Output Range**

Check what your model actually outputs:

**Add debug prints in `emotion_recognition_app.py` after prediction:**
```python
# Around line 424
predictions = recognizer.predict_emotion(recognizer.frame_buffer)

# ADD THIS:
if predictions is not None:
    print(f"Frame {frame_count}:")
    for i, emotion in enumerate(recognizer.emotions):
        print(f"  {emotion}: {predictions[i]:.3f}")
    print()
```

**Expected output:**
```
Frame 60:
  Boredom: 0.234
  Confusion: 0.867
  Frustration: 0.045
  Engagement: 0.123

Frame 90:
  Boredom: 0.456
  Confusion: 0.234
  Frustration: 0.123
  Engagement: 0.567  ← Should change over time!
```

**Bad output (what you might be seeing):**
```
Frame 60:
  Boredom: 0.12
  Confusion: 0.85  ← Always high
  Frustration: 0.03
  Engagement: 0.10

Frame 90:
  Boredom: 0.13
  Confusion: 0.87  ← Still high, others low
  Frustration: 0.02
  Engagement: 0.11
```

---

### 4. **Test with Different Expressions**

Record short test videos (5-10 seconds each) where you:
- **Smile widely** (should be Engagement)
- **Frown deeply** (should be Frustration or Boredom)
- **Look puzzled** (should be Confusion)
- **Blank stare** (should be Boredom)

Process each and check if model responds differently.

---

### 5. **Check Data Preprocessing**

Your model expects ResNet50 features. Verify:

**Is preprocessing correct?**
```python
# In emotion_recognition_app.py
frames_preprocessed = frames * 255.0  # 0-1 → 0-255
frames_preprocessed = preprocess_input(frames_preprocessed)  # ResNet50 preprocessing
```

**Does training match inference?**
- Training: Did you use the same ResNet50 preprocessing?
- Input size: Are frames 224×224 as expected?
- Normalization: Same as training?

---

### 6. **Retrain with Better Data**

If model is fundamentally broken, you may need to retrain:

**Tips for retraining:**
- Balance all 4 emotions (25% each)
- Include variety of:
  - Different people
  - Different lighting
  - Different angles
  - Different intensities
- Augment data (flip, brightness, contrast)
- Use validation set to check for overfitting

---

## 🎯 Quick Test Script

Add this to test your model's variety:

**Create `test_model_variety.py`:**
```python
import numpy as np
from emotion_recognition_app import AcademicEmotionRecognizer

# Load model
recognizer = AcademicEmotionRecognizer("all_emotions_final.h5")

# Create random fake data
print("Testing with random data (should give varied predictions):")
for i in range(5):
    # Random 60 frames
    random_frames = [np.random.rand(224, 224, 3).astype('float32') for _ in range(60)]
    
    # Fill buffer
    recognizer.frame_buffer.clear()
    for frame in random_frames:
        recognizer.frame_buffer.append(frame)
    
    # Predict
    predictions = recognizer.predict_emotion(recognizer.frame_buffer)
    emotion, confidence = recognizer.get_dominant_emotion(predictions)
    
    print(f"\nTest {i+1}:")
    print(f"  Dominant: {emotion} ({confidence:.3f})")
    print(f"  All scores:", {recognizer.emotions[j]: f"{predictions[j]:.3f}" for j in range(4)})

print("\n" + "="*60)
print("If all predictions are 'Confusion' with 0.85+ confidence,")
print("your model is likely broken or overfitted.")
print("="*60)
```

Run it:
```bash
python test_model_variety.py
```

**Good result:** Different emotions with varying confidences
**Bad result:** Always "Confusion 0.85"

---

## 📊 Understanding the Threshold

**How it works:**
```python
emotion_change_threshold = 0.6

# Logs ONLY when:
# 1. Confidence >= 0.6 AND
# 2. Emotion is different from previous
```

**If confusion is always 0.85:**
- First frame: Logs "Confusion 0.85" ✓
- Rest of frames: "Confusion 0.85" again → NOT logged (same emotion)

**That's why you only see one entry!**

---

## 🔧 Temporary Workaround

If you can't retrain immediately, you can force logging at intervals:

**Edit around line 424 in `emotion_recognition_app.py`:**
```python
if predictions is not None:
    emotion, confidence = recognizer.get_dominant_emotion(predictions)
    
    # Log every 30 frames OR when emotion changes
    should_log = (frame_count % 30 == 0) or \
                 (confidence >= recognizer.emotion_change_threshold and 
                  (current_emotion_state is None or emotion != current_emotion_state))
    
    if should_log:
        if current_emotion_state != emotion:
            current_emotion_state = emotion
        
        timestamp = recognizer.format_timestamp(frame_count)
        emotion_timeline.append({...})
```

This will log every second even if emotion doesn't change, giving you a complete timeline.

---

## ✅ Summary

**Most likely cause:** Model is overfitted or not properly trained

**Quick fixes:**
1. Lower threshold to 0.3-0.4
2. Test with exaggerated expressions
3. Add debug prints to see actual predictions

**Long-term fix:**
- Retrain model with balanced, diverse data
- Ensure preprocessing matches training

Good luck! 🎓
