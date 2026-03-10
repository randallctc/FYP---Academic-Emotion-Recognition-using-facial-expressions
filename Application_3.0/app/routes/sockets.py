"""
routes/sockets.py — All SocketIO event handlers.

Adding or modifying a real-time event means touching only this file.
Handlers are kept thin: parse the event data, call model/service methods,
emit responses. No ML or persistence logic lives here.
"""

import base64

import cv2
import numpy as np
from flask import request
from flask_socketio import join_room
from datetime import datetime

from app.models import SessionManager, Student
from app.ml.face import extract_face_crop

# Injected at registration time (see app/__init__.py).
_mgr      = None
_socketio = None


def init_handlers(mgr: SessionManager, socketio) -> None:
    global _mgr, _socketio
    _mgr      = mgr
    _socketio = socketio
    _register(socketio)


def _register(socketio) -> None:
    """Attach all event handlers to the SocketIO instance."""

    # ── Connection lifecycle ──────────────────────────────────────────────────

    @socketio.on("connect")
    def on_connect():
        print(f"[Socket] Connected: {request.sid}")

    @socketio.on("disconnect")
    def on_disconnect():
        sid = request.sid

        # Try teacher first.
        room_code = _mgr.remove_teacher(sid)
        if room_code:
            print(f"[Socket] Teacher disconnected from room {room_code}")
            socketio.emit(
                "teacher_disconnected",
                {"message": "Teacher has ended the session"},
                room=room_code,
            )
            return

        # Try student.
        room_code, name = _mgr.remove_student(sid)
        if room_code:
            print(f"[Socket] Student disconnected from room {room_code}: {name}")
            _broadcast_student_list(room_code)
            _push_aggregate(room_code)
        else:
            print(f"[Socket] Unknown socket disconnected: {sid}")

    # ── Room join ─────────────────────────────────────────────────────────────

    @socketio.on("join_room")
    def on_join_room(data):
        room_code  = data.get("room_code")
        is_teacher = data.get("is_teacher", False)
        room       = _mgr.get_room(room_code)
        if not room:
            return

        join_room(room_code)

        if is_teacher:
            _mgr.set_teacher_socket(room_code, request.sid)
            print(f"[Socket] Teacher joined room {room_code}")
        else:
            print(f"[Socket] Socket joined room {room_code}")

    # ── Student events ────────────────────────────────────────────────────────

    @socketio.on("student_join")
    def on_student_join(data):
        student_id = data.get("student_id")
        name       = data.get("student_name", "Unknown")
        room_code  = data.get("room_code")

        room = _mgr.get_room(room_code)
        if not room:
            print(f"[Socket] student_join: invalid room {room_code}")
            return

        student = Student(student_id, name, request.sid)
        _mgr.add_student(room_code, student)

        print(f"[Socket] Student joined room {room_code}: {name} ({student_id})")
        _broadcast_student_list(room_code)

    @socketio.on("student_leave")
    def on_student_leave(data):
        student_id = data.get("student_id")
        room_code  = data.get("room_code")
        room       = _mgr.get_room(room_code)
        if not room:
            return

        student = room.students.get(student_id)
        if student:
            name = student.name
            # Use remove_student to keep all maps in sync.
            _mgr.remove_student(student.socket_id)
            print(f"[Socket] Student left room {room_code}: {name}")
            _broadcast_student_list(room_code)
            _push_aggregate(room_code)

    @socketio.on("student_frame")
    def on_student_frame(data):
        student_id = data.get("student_id")
        room_code  = data.get("room_code")
        frame_data = data.get("frame")

        room = _mgr.get_room(room_code)
        if not room or student_id not in room.students:
            return

        frame  = _decode_image(frame_data)
        crop   = extract_face_crop(frame)
        student = room.students[student_id]

        if crop is None:
            socketio.emit("buffer_update", {
                "student_id":  student_id,
                "buffer_size": len(student.buffer),
                "buffer_full": len(student.buffer) >= 60,
                "face_found":  False,
            }, room=room_code)
            return

        # Rolling buffer: drop oldest frame when full.
        if len(student.buffer) >= 60:
            student.buffer.pop(0)
        student.buffer.append(crop)
        student.last_face_time = datetime.now()

        socketio.emit("buffer_update", {
            "student_id":  student_id,
            "buffer_size": len(student.buffer),
            "buffer_full": len(student.buffer) >= 60,
            "face_found":  True,
        }, room=room_code)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _decode_image(b64_string: str) -> np.ndarray | None:
    """Decode a base64 image string (with or without data-URL prefix)."""
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        arr      = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _broadcast_student_list(room_code: str) -> None:
    room = _mgr.get_room(room_code)
    if not room:
        return
    _socketio.emit("student_list_update", {
        "students": [s.to_dict() for s in room.students.values()],
        "count":    len(room.students),
    }, room=room_code)


def _push_aggregate(room_code: str) -> None:
    """
    Immediately push a classroom_update after a student leaves so the
    teacher dashboard doesn't wait for the next scheduler cycle.
    """
    from app.services.recording import elapsed_str
    room = _mgr.get_room(room_code)
    if not room:
        return

    if room.students:
        from app.services.scheduler import _aggregate_emotions
        agg = _aggregate_emotions(room)
        if agg:
            _socketio.emit("classroom_update", agg, room=room_code)
    else:
        _socketio.emit("classroom_update", {
            "timestamp":      elapsed_str(room.start_time),
            "total_students": 0,
            "emotions":       {},
            "percentages":    {},
            "alert":          None,
        }, room=room_code)
