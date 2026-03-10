"""
ml/face.py — Face detection and cropping.

Responsibilities:
  - Detect the largest face in a BGR frame.
  - Crop with margin, resize to FRAME_SIZE, normalise to [0, 1].

Not responsible for: model inference, Flask, sockets.
"""

import cv2
import numpy as np

from app.config import FRAME_SIZE, MARGIN_RATIO

# Load cascade once at import time (shared across all callers).
_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face_crop(frame: np.ndarray) -> np.ndarray | None:
    """
    Detect the largest face in `frame`, crop it with margin, resize
    to (FRAME_SIZE, FRAME_SIZE), and normalise pixel values to [0, 1].

    Args:
        frame: BGR image as a numpy array (H, W, 3).

    Returns:
        float32 array of shape (FRAME_SIZE, FRAME_SIZE, 3), or None if
        no face is detected or the crop is empty.
    """
    if frame is None:
        return None

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]

    # Expand bounding box by MARGIN_RATIO on each side.
    mx = int(w * MARGIN_RATIO)
    my = int(h * MARGIN_RATIO)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))
    return crop.astype(np.float32) / 255.0
