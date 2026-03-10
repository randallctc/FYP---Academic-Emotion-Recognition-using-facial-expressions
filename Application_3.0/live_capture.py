"""
live_capture.py — Standalone single-camera emotion recognition tool.

Run directly:
    python live_capture.py
    python live_capture.py path/to/model.h5

Controls:
    R   — Start / stop recording
    ESC — Exit

This file is completely independent of the classroom server.
It imports only the ML modules (EmotionPredictor, extract_face_crop)
and opencv for display.
"""

import json
import os
import sys
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from app.config    import BUFFER_SIZE, FRAME_SIZE
from app.ml.face   import extract_face_crop
from app.ml.predictor import EmotionPredictor

# ── Config ────────────────────────────────────────────────────────────────────
CAMERA_ID           = 0
CAPTURE_WIDTH       = 1280
CAPTURE_HEIGHT      = 720
FPS                 = 30
PREDICTION_INTERVAL = 30   # frames between predictions
CONFIDENCE_THRESHOLD = 0.6  # minimum confidence to log an emotion change


# ── Live capture loop ─────────────────────────────────────────────────────────

def run(predictor: EmotionPredictor) -> None:
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    buffer:        deque = deque(maxlen=BUFFER_SIZE)
    last_preds:    dict  = {}
    is_recording   = False
    video_writer   = None
    output_file    = None
    emotion_log:   list  = []
    last_emotion:  str   = ""
    frame_count    = 0
    frames_since_pred = 0

    print("\n=== Live Emotion Capture ===")
    print("R — Start/stop recording | ESC — Exit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        crop    = extract_face_crop(frame)

        if crop is not None:
            x, y, w, h = _find_face_bbox(frame)
            if x is not None:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            buffer.append(crop)
            frames_since_pred += 1

            if len(buffer) == BUFFER_SIZE and frames_since_pred >= PREDICTION_INTERVAL:
                last_preds        = predictor.predict(list(buffer)) or {}
                frames_since_pred = 0

                if is_recording and last_preds:
                    emotion, conf = predictor.dominant_emotion(last_preds)
                    if conf >= CONFIDENCE_THRESHOLD and emotion != last_emotion:
                        ts = _fmt_ts(frame_count, FPS)
                        emotion_log.append({
                            "timestamp":  ts,
                            "frame":      frame_count,
                            "emotion":    emotion,
                            "confidence": round(conf, 3),
                            "all_scores": last_preds,
                        })
                        last_emotion = emotion
                        print(f"[{ts}] {emotion} ({conf:.2f})")

        display = _draw_ui(display, buffer, last_preds, predictor, is_recording)

        if is_recording and video_writer:
            video_writer.write(display)
            frame_count += 1

        cv2.imshow("Live Emotion Capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key in (ord("r"), ord("R")):
            if not is_recording:
                video_writer, output_file = _start_recording(FPS, CAPTURE_WIDTH, CAPTURE_HEIGHT)
                is_recording = True
                emotion_log  = []
                frame_count  = 0
                last_emotion = ""
            else:
                is_recording = _stop_recording(video_writer, output_file, emotion_log)

    if is_recording:
        _stop_recording(video_writer, output_file, emotion_log)
    cap.release()
    cv2.destroyAllWindows()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_face_bbox(frame):
    """Return (x, y, w, h) of the largest detected face, or (None, ...) if none."""
    import cv2 as _cv
    cascade = _cv.CascadeClassifier(
        _cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray  = _cv.cvtColor(frame, _cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None, None, None
    return faces[0]


def _draw_ui(frame, buffer, preds, predictor, is_recording):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 260), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if is_recording:
        cv2.circle(frame, (30, 40), 10, (0, 0, 255), -1)
        cv2.putText(frame, "RECORDING", (50, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Buffer: {len(buffer)}/{BUFFER_SIZE}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if preds:
        y = 120
        for emotion, score in preds.items():
            thresh = predictor.thresholds.get(emotion, 0.5)
            color  = (0, 255, 0) if score >= thresh else (255, 255, 255)
            bar_w  = int(200 * score)
            cv2.putText(frame, emotion, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (150, y - 15), (150 + bar_w, y - 5), color, -1)
            cv2.rectangle(frame, (150, y - 15), (350, y - 5), (100, 100, 100), 1)
            cv2.putText(frame, f"{score:.2f}", (360, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y += 30

    h = frame.shape[0]
    cv2.putText(frame, "R - Start/Stop Recording | ESC - Exit",
                (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def _fmt_ts(frame_number: int, fps: int) -> str:
    secs = frame_number / fps
    m, s = divmod(int(secs), 60)
    ms   = int((secs % 1) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _start_recording(fps, width, height):
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"emotion_recording_{ts}.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    print(f"Recording started: {filename}")
    return writer, filename


def _stop_recording(writer, filename, emotion_log) -> bool:
    if writer:
        writer.release()
    if filename:
        json_file = filename.replace(".mp4", "_annotations.json")
        with open(json_file, "w") as f:
            json.dump(emotion_log, f, indent=2)
        print(f"Recording stopped: {filename}")
        print(f"Annotations saved: {json_file}")
    return False   # is_recording = False


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "final_model_v2.h5"
    predictor  = EmotionPredictor(model_path)
    run(predictor)
