"""
services/recording.py — Session data builder.

Responsibilities:
  - Build a completed session report dict from a Room.
  - Optionally save it to a local JSON file (local dev only).

On Render (cloud), saving to disk is skipped — the report dict is returned
directly in the HTTP response and the teacher's browser downloads it as a
local file. No server-side persistence needed.
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


def build_session_report(room) -> dict:
    """
    Build and return the full session report as a dict.
    Does not touch the filesystem.
    """
    return {
        "room_code":         room.room_code,
        "session_id":        room.session_id,
        "teacher_name":      room.teacher_name,
        "start_time":        room.start_time.isoformat() if room.start_time else None,
        "duration":          elapsed_str(room.start_time),
        "total_students":    len(room.students),
        "students":          [s.to_dict() for s in room.students.values()],
        "timeline":          room.emotion_timeline,
        "total_data_points": len(room.emotion_timeline),
    }


def save_session(room) -> str:
    """
    Save a session report to a local JSON file.
    Returns the path of the saved file.
    Used in local development only.
    """
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    filename = f"room_{room.room_code}_session_{room.session_id}.json"
    path     = os.path.join(SESSIONS_DIR, filename)

    with open(path, "w") as f:
        json.dump(build_session_report(room), f, indent=2)

    return path


def list_sessions() -> list[dict]:
    """Return summary dicts for all locally saved sessions, newest first."""
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
    Find and return the full session JSON for a given session_id.
    Returns None if not found. Used in local development only.
    """
    if not os.path.exists(SESSIONS_DIR):
        return None

    for fname in os.listdir(SESSIONS_DIR):
        if session_id in fname and fname.endswith(".json"):
            with open(os.path.join(SESSIONS_DIR, fname)) as f:
                return json.load(f)

    return None
