"""
models.py — In-memory data models for rooms and students.

These are plain Python classes with no Flask or ML dependencies.
Adding a new field to a student or room means touching only this file.
"""

import random
import threading
from datetime import datetime


class Student:
    """Represents one connected student in a room."""

    def __init__(self, student_id: str, name: str, socket_id: str):
        self.student_id = student_id
        self.name       = name
        self.socket_id  = socket_id
        self.join_time  = datetime.now().isoformat()
        self.connected  = True

        # Frame buffer — list of face crops (float32, 224x224x3)
        self.buffer: list = []

        # Latest prediction results.
        # emotion is None until the first real prediction completes —
        # students with emotion=None are excluded from the classroom aggregate.
        self.emotion:    str | None = None
        self.confidence: float      = 0.0
        self.all_scores: dict       = {}

        # True once at least one prediction has been emitted for this student.
        self.has_prediction: bool = False

        # Guard flag: prevents concurrent predictions for the same student.
        self.predicting: bool = False

        # Timestamp of the last frame where a face was successfully detected.
        # Used by the scheduler to detect when a student has left the camera.
        # None means no face has ever been seen.
        self.last_face_time: datetime | None = None

        # True once the student has been absent for >= ABSENT_SECONDS.
        # Prediction is skipped while True.
        self.is_absent: bool = False

        # Set when a face is re-detected after being absent.
        # Prediction is held for REANALYZE_SECONDS after this timestamp
        # to let fresh frames fill the buffer before running inference.
        self.face_returned_time: datetime | None = None

    def to_dict(self) -> dict:
        """Serialisable snapshot (excludes the buffer)."""
        return {
            "id":             self.student_id,
            "name":           self.name,
            "connected":      self.connected,
            "join_time":      self.join_time,
            "emotion":        self.emotion,
            "confidence":     self.confidence,
            "has_prediction": self.has_prediction,
        }


class Room:
    """Represents one active classroom session."""

    def __init__(self, room_code: str, teacher_name: str):
        self.room_code    = room_code
        self.teacher_name = teacher_name
        self.created_at   = datetime.now()

        # Socket IDs
        self.teacher_socket: str | None = None

        # Student registry
        self.students:          dict[str, Student] = {}   # student_id → Student
        self.socket_to_student: dict[str, str]     = {}   # socket_id  → student_id

        # Recording state
        self.is_recording    = False
        self.start_time:  datetime | None = None
        self.session_id:  str      | None = None
        self.emotion_timeline: list = []


class SessionManager:
    """
    Global registry of all active rooms and socket mappings.

    All mutations are protected by a threading.Lock so the prediction
    scheduler thread and the SocketIO event thread don't race.
    """

    def __init__(self):
        self.rooms:          dict[str, Room] = {}   # room_code → Room
        self.socket_to_room: dict[str, str]  = {}   # socket_id → room_code
        self._lock = threading.Lock()

    # ── Room management ───────────────────────────────────────────────────────

    def create_room(self, teacher_name: str) -> str:
        """Generate a unique 6-digit room code and register the room."""
        with self._lock:
            while True:
                code = "".join(str(random.randint(0, 9)) for _ in range(6))
                if code not in self.rooms:
                    self.rooms[code] = Room(code, teacher_name)
                    return code

    def get_room(self, room_code: str) -> Room | None:
        return self.rooms.get(room_code)

    # ── Student management ────────────────────────────────────────────────────

    def add_student(self, room_code: str, student: Student) -> None:
        """Register a student and update all socket maps."""
        with self._lock:
            room = self.rooms[room_code]
            # Clean up any previous socket mapping for this student_id (reconnect).
            existing = room.students.get(student.student_id)
            if existing:
                room.socket_to_student.pop(existing.socket_id, None)
                self.socket_to_room.pop(existing.socket_id, None)

            room.students[student.student_id]       = student
            room.socket_to_student[student.socket_id] = student.student_id
            self.socket_to_room[student.socket_id]    = room_code

    def remove_student(self, socket_id: str) -> tuple[str | None, str | None]:
        """
        Remove a student by socket_id.
        Returns (room_code, student_name) or (None, None) if not found.
        """
        with self._lock:
            room_code = self.socket_to_room.get(socket_id)
            if not room_code:
                return None, None
            room = self.rooms.get(room_code)
            if not room:
                return None, None
            student_id = room.socket_to_student.get(socket_id)
            if not student_id:
                return None, None

            student = room.students.pop(student_id, None)
            room.socket_to_student.pop(socket_id, None)
            self.socket_to_room.pop(socket_id, None)
            return room_code, (student.name if student else None)

    # ── Teacher management ────────────────────────────────────────────────────

    def set_teacher_socket(self, room_code: str, socket_id: str) -> None:
        with self._lock:
            room = self.rooms.get(room_code)
            if room:
                room.teacher_socket = socket_id
                self.socket_to_room[socket_id] = room_code

    def remove_teacher(self, socket_id: str) -> str | None:
        """
        Detach a teacher by socket_id.
        Returns room_code or None if not found.
        """
        with self._lock:
            for room_code, room in self.rooms.items():
                if room.teacher_socket == socket_id:
                    room.teacher_socket = None
                    self.socket_to_room.pop(socket_id, None)
                    return room_code
            return None
