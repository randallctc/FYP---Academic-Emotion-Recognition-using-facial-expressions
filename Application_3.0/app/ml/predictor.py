"""
ml/predictor.py — Emotion inference pipeline.

Responsibilities:
  - Load and hold the ResNet50 feature extractor and BiGRU classifier.
  - Extract features from a buffer of face crops.
  - Return per-emotion sigmoid scores as a plain dict.
  - Apply per-emotion thresholds to determine the dominant emotion.

Not responsible for: face detection, camera capture, Flask, sockets.
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from app.config import EMOTIONS, MODEL_PATH, THRESHOLD_PATH, BUFFER_SIZE, FRAME_SIZE

# ── Keras GRU compatibility patch ─────────────────────────────────────────────
# Older saved models may include a `time_major` argument that was removed.
# This patch silently drops it on load so the model opens without errors.
import keras
from keras.layers import GRU as _GRU

_orig_gru_init = _GRU.__init__

def _patched_gru_init(self, *args, **kwargs):
    kwargs.pop("time_major", None)
    _orig_gru_init(self, *args, **kwargs)

_GRU.__init__ = _patched_gru_init
# ─────────────────────────────────────────────────────────────────────────────


class EmotionPredictor:
    """
    Wraps the ResNet50 + BiGRU inference pipeline.

    Usage:
        predictor = EmotionPredictor()
        scores    = predictor.predict(buffer)   # buffer = list of 60 face crops
        emotion, confidence = predictor.dominant_emotion(scores)
    """

    def __init__(
        self,
        model_path:     str = MODEL_PATH,
        threshold_path: str = THRESHOLD_PATH,
    ):
        self._load_gru(model_path)
        self._load_resnet()
        self._load_thresholds(threshold_path)

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _load_gru(self, path: str):
        print(f"[Predictor] Loading GRU model: {path}")
        self.gru = tf.keras.models.load_model(path, compile=False)
        print("[Predictor] GRU model loaded.")

    def _load_resnet(self):
        print("[Predictor] Loading ResNet50...")
        self.resnet = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(FRAME_SIZE, FRAME_SIZE, 3),
        )
        self.resnet.trainable = False
        print("[Predictor] ResNet50 loaded.")

    def _load_thresholds(self, path: str):
        if os.path.exists(path):
            with open(path) as f:
                self.thresholds = json.load(f)
            print(f"[Predictor] Thresholds loaded: {self.thresholds}")
        else:
            print(f"[Predictor] WARNING: {path} not found — using 0.5 for all emotions.")
            self.thresholds = {e: 0.5 for e in EMOTIONS}

    # ── Inference ─────────────────────────────────────────────────────────────

    def extract_features(self, buffer: list) -> np.ndarray | None:
        """
        Convert a buffer of 60 face crops (float32, [0,1]) into a
        (60, 2048) ResNet50 feature array.

        Returns None if the buffer is not full.
        """
        if len(buffer) < BUFFER_SIZE:
            return None

        frames = np.array(buffer)             # (60, 224, 224, 3)
        frames = frames * 255.0               # [0,1] → [0,255]
        frames = preprocess_input(frames)     # caffe-style normalisation
        return self.resnet.predict(frames, verbose=0, batch_size=BUFFER_SIZE)

    def predict(self, buffer: list) -> dict | None:
        """
        Run the full inference pipeline on a buffer of face crops.

        Returns:
            dict  {emotion: float score}  — raw sigmoid values in [0, 1]
            None  — if the buffer is not full yet
        """
        features = self.extract_features(buffer)
        if features is None:
            return None

        features = np.expand_dims(features, axis=0)   # (1, 60, 2048)
        raw = self.gru.predict(features, verbose=0)

        # Multi-output model returns a dict; single-output returns an array.
        if isinstance(raw, dict):
            return {e: float(np.squeeze(raw[e])) for e in EMOTIONS}
        else:
            raw = np.squeeze(raw)
            return {e: float(raw[i]) for i, e in enumerate(EMOTIONS)}

    def dominant_emotion(self, scores: dict) -> tuple[str, float]:
        """
        Apply per-emotion thresholds to the raw scores.

        Returns:
            (emotion_name, score)  for the highest scorer above its threshold.
            ('Neutral', 0.0)       if nothing clears any threshold.
        """
        if not scores:
            return "Neutral", 0.0

        above = {e: s for e, s in scores.items() if s >= self.thresholds.get(e, 0.5)}

        if above:
            best = max(above, key=above.get)
            return best, scores[best]

        return "Neutral", 0.0
