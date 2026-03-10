"""
services/scheduler.py — Background prediction loop.

Runs every PREDICTION_INTERVAL seconds, iterates over all rooms and
students, runs inference on any student with a full buffer, then
broadcasts the aggregated classroom state to the teacher.

Keeping this in its own file means the prediction logic can be tuned
or replaced (e.g. swapped for async tasks) without touching routes.
"""

import time
import threading
from collections import defaultdict
from datetime import datetime

from app.config import (
    EMOTIONS,
    PREDICTION_INTERVAL,
    ALERT_BOREDOM_PCT,
    ALERT_CONFUSION_PCT,
    ALERT_LOW_ENGAGE_PCT,
    BUFFER_SIZE,
    ABSENT_SECONDS,
    REANALYZE_SECONDS,
)
from app.services.recording import elapsed_str


def _aggregate_emotions(room) -> dict | None:
    """
    Build the classroom_update payload from current student states.

    Only students who have received at least one real prediction are counted.
    Students still in the "analyzing" phase (has_prediction=False) or whose
    emotion has been cleared due to absence are excluded from the aggregate.

    Returns None if no students have a confirmed emotion yet.
    """
    confirmed = [s for s in room.students.values() if s.has_prediction and s.emotion is not None]
    if not confirmed:
        return None

    counts = defaultdict(int)
    for s in confirmed:
        counts[s.emotion] += 1

    total = len(confirmed)
    pct   = {e: counts[e] / total * 100 for e in counts}

    # Ensure every emotion key is present (so the frontend doesn't have to guard).
    for e in EMOTIONS + ["Neutral"]:
        pct.setdefault(e, 0.0)
        counts.setdefault(e, 0)

    # Build alert string if a threshold is breached.
    alert = None
    if pct.get("Boredom", 0) > ALERT_BOREDOM_PCT:
        alert = f"High boredom: {pct['Boredom']:.1f}% of students"
    elif pct.get("Confusion", 0) > ALERT_CONFUSION_PCT:
        alert = f"High confusion: {pct['Confusion']:.1f}% of students"
    elif (pct.get("Boredom", 0) + pct.get("Confusion", 0) + pct.get("Frustration", 0)) > ALERT_LOW_ENGAGE_PCT:
        alert = f"Low engagement: only {pct.get('Engagement', 0):.1f}% engaged"

    return {
        "timestamp":      elapsed_str(room.start_time),
        "total_students": len(room.students),   # total connected, not just confirmed
        "confirmed":      total,                # students with a live emotion
        "tracked":        total,                # alias used in review page display
        "emotions":       dict(counts),
        "percentages":    pct,
        "alert":          alert,
    }


def _empty_update(room) -> dict:
    """Payload sent when a room becomes empty."""
    return {
        "timestamp":      elapsed_str(room.start_time),
        "total_students": 0,
        "emotions":       {},
        "percentages":    {},
        "alert":          None,
    }


def start_scheduler(mgr, predictor, socketio) -> threading.Thread:
    """
    Start the background prediction thread and return it.

    Args:
        mgr:       SessionManager  — shared room registry.
        predictor: EmotionPredictor — inference engine.
        socketio:  SocketIO         — for emitting events.
    """

    def _loop():
        print("[Scheduler] Started.")
        while True:
            time.sleep(PREDICTION_INTERVAL)
            try:
                _run_cycle(mgr, predictor, socketio)
            except Exception as e:
                print(f"[Scheduler] Unhandled error: {e}")
                import traceback; traceback.print_exc()

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return thread


def _run_cycle(mgr, predictor, socketio) -> None:
    """One prediction cycle across all rooms."""
    for room_code, room in list(mgr.rooms.items()):

        if not room.students:
            # Room is empty — send a zeroed update so the teacher dashboard clears.
            socketio.emit("classroom_update", _empty_update(room), room=room_code)
            continue

        predicted_any = False

        for student_id, student in list(room.students.items()):
            now = datetime.now()

            # ── Absent → Away transition ──────────────────────────────────────
            # Once last_face_time is set, track how long since the last face.
            if student.last_face_time is not None:
                seconds_since_face = (now - student.last_face_time).total_seconds()

                if not student.is_absent and seconds_since_face >= ABSENT_SECONDS:
                    # Just crossed the absent threshold — mark as away.
                    student.is_absent        = True
                    student.face_returned_time = None
                    student.emotion          = None
                    student.confidence       = 0.0
                    student.all_scores       = {}
                    print(f"[{room_code}] {student.name}: away from camera")
                    socketio.emit("emotion_update", {
                        "student_id": student_id,
                        "emotion":    None,
                        "confidence": 0.0,
                        "all_scores": {},
                        "status":     "absent",
                    }, room=room_code)
                    socketio.emit("student_emotion_update", {
                        "student_id":     student_id,
                        "emotion":        None,
                        "confidence":     0.0,
                        "has_prediction": True,
                    }, room=room_code)

                elif student.is_absent and seconds_since_face < ABSENT_SECONDS:
                    # Face returned — start the re-analyze window.
                    student.is_absent = False
                    student.face_returned_time = now
                    print(f"[{room_code}] {student.name}: face returned, re-analyzing")
                    socketio.emit("emotion_update", {
                        "student_id": student_id,
                        "emotion":    None,
                        "confidence": 0.0,
                        "all_scores": {},
                        "status":     "analyzing",
                    }, room=room_code)

            # ── Skip prediction if absent ─────────────────────────────────────
            if student.is_absent:
                continue

            # ── Skip prediction during re-analyze window ──────────────────────
            if student.face_returned_time is not None:
                seconds_since_return = (now - student.face_returned_time).total_seconds()
                # Also verify face is still being seen (last_face_time is recent)
                face_is_recent = (
                    student.last_face_time is not None and
                    (now - student.last_face_time).total_seconds() < ABSENT_SECONDS
                )
                if seconds_since_return < REANALYZE_SECONDS or not face_is_recent:
                    continue  # still collecting fresh frames, hold off
                else:
                    student.face_returned_time = None  # window elapsed, predict normally

            if len(student.buffer) < BUFFER_SIZE or student.predicting:
                continue

            student.predicting = True
            try:
                scores = predictor.predict(student.buffer)
                if scores is None:
                    continue

                emotion, confidence = predictor.dominant_emotion(scores)

                student.emotion        = emotion
                student.confidence     = float(confidence)
                student.all_scores     = {e: round(float(s), 3) for e, s in scores.items()}
                student.has_prediction = True

                print(f"[{room_code}] {student.name}: {emotion} "
                      f"({student.confidence:.2f}) | {student.all_scores}")

                # Notify the student's own page.
                socketio.emit("emotion_update", {
                    "student_id": student_id,
                    "emotion":    emotion,
                    "confidence": student.confidence,
                    "all_scores": student.all_scores,
                    "status":     "detected",
                }, room=room_code)

                # Notify the teacher's student list panel.
                socketio.emit("student_emotion_update", {
                    "student_id":     student_id,
                    "emotion":        emotion,
                    "confidence":     student.confidence,
                    "has_prediction": True,
                }, room=room_code)

                predicted_any = True

            except Exception as e:
                print(f"[Scheduler] Prediction error for {student.name}: {e}")
                import traceback; traceback.print_exc()
            finally:
                student.predicting = False

        # Build aggregate — may be None if no students have a confirmed emotion.
        agg = _aggregate_emotions(room)

        # Always broadcast something so the teacher dashboard stays current.
        if agg:
            socketio.emit("classroom_update", agg, room=room_code)
        else:
            # Students are connected but none are confirmed (all away or analyzing).
            fallback = {
                "timestamp":      elapsed_str(room.start_time),
                "total_students": len(room.students),
                "confirmed":      0,
                "tracked":        0,
                "emotions":       {},
                "percentages":    {},
                "alert":          None,
            }
            socketio.emit("classroom_update", fallback, room=room_code)
            agg = fallback   # use for timeline recording below

        if room.is_recording:
            # Always record a timeline entry — gaps where all students are away
            # should show as "0 of N tracked" in the review, not disappear.
            room.emotion_timeline.append(agg)
            if agg.get("alert"):
                socketio.emit("recording_alert", {
                    "timestamp": agg["timestamp"],
                    "alert":     agg["alert"],
                }, room=room_code)
