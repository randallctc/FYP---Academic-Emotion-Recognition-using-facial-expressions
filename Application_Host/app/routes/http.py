"""
routes/http.py — All HTTP route handlers.

Adding a new page or API endpoint means touching only this file.
Each route is kept thin: validate input, delegate to a service, return JSON.
"""

import base64

import cv2
import numpy as np
from flask import Blueprint, jsonify, render_template, request

from app.models import SessionManager
from app.services import recording as rec_service

# Blueprint is registered on the Flask app in app/__init__.py
http_bp = Blueprint("http", __name__)

# Injected at registration time (see app/__init__.py).
_mgr: SessionManager = None


def init_routes(mgr: SessionManager) -> None:
    global _mgr
    _mgr = mgr


# ── Pages ─────────────────────────────────────────────────────────────────────

@http_bp.route("/")
def home():
    return render_template("home.html")


@http_bp.route("/teacher/<room_code>")
def teacher_dashboard(room_code):
    if not _mgr.get_room(room_code):
        return "Room not found", 404
    return render_template("teacher.html", room_code=room_code)


@http_bp.route("/student/<room_code>")
def student_page(room_code):
    if not _mgr.get_room(room_code):
        return "Room not found", 404
    return render_template("student.html", room_code=room_code)


@http_bp.route("/review")
def review_page():
    return render_template("review.html")


# ── Room API ──────────────────────────────────────────────────────────────────

@http_bp.route("/api/create_room", methods=["POST"])
def create_room():
    data         = request.json or {}
    teacher_name = data.get("teacher_name", "").strip()
    if not teacher_name:
        return jsonify({"success": False, "error": "Teacher name required"})
    code = _mgr.create_room(teacher_name)
    print(f"[HTTP] Room created: {code} by {teacher_name}")
    return jsonify({"success": True, "room_code": code})


@http_bp.route("/api/verify_room", methods=["POST"])
def verify_room():
    data = request.json or {}
    code = data.get("room_code", "").strip()
    room = _mgr.get_room(code)
    if room:
        return jsonify({"valid": True, "teacher_name": room.teacher_name})
    return jsonify({"valid": False})


# ── Session / Recording API ───────────────────────────────────────────────────

@http_bp.route("/api/sessions")
def list_sessions():
    return jsonify(rec_service.list_sessions())


@http_bp.route("/analytics/<session_id>")
def get_analytics(session_id):
    data = rec_service.load_session(session_id)
    if data:
        return jsonify(data)
    return jsonify({"error": "Session not found"}), 404


@http_bp.route("/start_recording", methods=["POST"])
def start_recording():
    from datetime import datetime
    from flask import current_app

    data      = request.json or {}
    room_code = data.get("room_code")
    room      = _mgr.get_room(room_code)
    if not room:
        return jsonify({"success": False, "error": "Invalid room"})

    room.is_recording     = True
    room.start_time       = datetime.now()
    room.session_id       = datetime.now().strftime("%Y%m%d_%H%M%S")
    room.emotion_timeline = []

    print(f"[HTTP] Recording started in room {room_code}: {room.session_id}")

    # Import socketio here to avoid circular imports.
    from app import socketio
    socketio.emit("clear_alerts",      room=room_code)
    socketio.emit("recording_started", {"session_id": room.session_id}, room=room_code)

    return jsonify({"success": True, "session_id": room.session_id})


@http_bp.route("/stop_recording", methods=["POST"])
def stop_recording():
    data      = request.json or {}
    room_code = data.get("room_code")
    room      = _mgr.get_room(room_code)
    if not room:
        return jsonify({"success": False, "error": "Invalid room"})
    if not room.is_recording:
        return jsonify({"success": False, "error": "Not recording"})

    room.is_recording = False
    report            = rec_service.build_session_report(room)
    filename          = f"room_{room.room_code}_session_{room.session_id}.json"
    data_points       = len(room.emotion_timeline)

    print(f"[HTTP] Recording stopped. {data_points} data points. Sending to teacher browser.")

    from app import socketio
    socketio.emit("recording_stopped", {
        "file":        filename,
        "data_points": data_points,
    }, room=room_code)

    # Return the full session report so the teacher's browser can download it
    # as a local file. Nothing is saved server-side (no persistent disk on Render).
    return jsonify({
        "success":     True,
        "file":        filename,
        "data_points": data_points,
        "report":      report,
    })
