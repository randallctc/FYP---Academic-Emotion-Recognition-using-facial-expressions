"""
services/recording.py — Session persistence.

Responsibilities:
  - Save a completed session to JSON.
  - List saved sessions.
  - Load a session by session_id for the review page.

Not responsible for: Flask routing, sockets, ML.
"""

import json
import os
from datetime import datetime

from app.config import SESSIONS_DIR


def elapsed_str(start_time: datetime | None) -> str:
    """Format elapsed time since start_time as MM:SS.mmm."""
    if not start_time:
        return "00:00.000"
    e  = (datetime.now() - start_time).total_seconds()
    m  = int(e // 60)
    s  = int(e % 60)
    ms = int((e % 1) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def save_session(room) -> str:
    """
    Serialise a Room's recording data to a JSON file.

    Returns the path of the saved file.
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    filename = f"room_{room.room_code}_session_{room.session_id}.json"
    path     = os.path.join(SESSIONS_DIR, filename)

    report = {
        "room_code":        room.room_code,
        "session_id":       room.session_id,
        "teacher_name":     room.teacher_name,
        "start_time":       room.start_time.isoformat() if room.start_time else None,
        "duration":         elapsed_str(room.start_time),
        "total_students":   len(room.students),
        "students":         [s.to_dict() for s in room.students.values()],
        "timeline":         room.emotion_timeline,
        "total_data_points": len(room.emotion_timeline),
    }

    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return path


def list_sessions() -> list[dict]:
    """Return summary dicts for all saved sessions, newest first."""
    if not os.path.exists(SESSIONS_DIR):
        return []

    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, fname)) as f:
                d = json.load(f)
            sessions.append({
                "id":          d["session_id"],
                "date":        d["session_id"][:8],
                "duration":    d.get("duration", "Unknown"),
                "students":    d.get("total_students", 0),
                "data_points": len(d.get("timeline", [])),
            })
        except Exception:
            pass

    sessions.sort(key=lambda x: x["id"], reverse=True)
    return sessions


def load_session(session_id: str) -> dict | None:
    """
    Find and return the full session JSON for the given session_id.
    Returns None if not found.
    """
    if not os.path.exists(SESSIONS_DIR):
        return None

    for fname in os.listdir(SESSIONS_DIR):
        if session_id in fname and fname.endswith(".json"):
            with open(os.path.join(SESSIONS_DIR, fname)) as f:
                return json.load(f)

    return None
