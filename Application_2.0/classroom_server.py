"""
Classroom Emotion Monitoring Server - Clean Rewrite
"""

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import numpy as np
import base64
from datetime import datetime
import json
import os
import threading
import time
from collections import defaultdict
from emotion_recognition_app import AcademicEmotionRecognizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'classroom_secret_2024'

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=120,
    ping_interval=25,
    async_mode='threading'
)

# Load model once at startup
print("Loading emotion recognition model...")
recognizer = AcademicEmotionRecognizer("final_model_v2.h5", "emotion_thresholds.json")
print("Model loaded successfully!")

# ── Data structures ────────────────────────────────────────────────────────────

class Student:
    def __init__(self, student_id, name, socket_id):
        self.student_id  = student_id
        self.name        = name
        self.socket_id   = socket_id
        self.buffer      = []           # list of face crops (float32, 224x224x3)
        self.emotion     = "Neutral"
        self.confidence  = 0.0
        self.all_scores  = {}
        self.connected   = True
        self.join_time   = datetime.now().isoformat()
        self.predicting  = False

class Room:
    def __init__(self, room_code, teacher_name):
        self.room_code          = room_code
        self.teacher_name       = teacher_name
        self.teacher_socket     = None
        self.students           = {}    # {student_id: Student}
        self.socket_to_student  = {}    # {socket_id: student_id}
        self.emotion_timeline   = []
        self.is_recording       = False
        self.start_time         = None
        self.session_id         = None
        self.created_at         = datetime.now()

class SessionManager:
    def __init__(self):
        self.rooms          = {}    # {room_code: Room}
        self.socket_to_room = {}    # {socket_id: room_code}
        self.lock           = threading.Lock()

    def create_room(self, teacher_name):
        import random
        with self.lock:
            while True:
                code = ''.join([str(random.randint(0, 9)) for _ in range(6)])
                if code not in self.rooms:
                    self.rooms[code] = Room(code, teacher_name)
                    return code

    def get_room(self, room_code):
        return self.rooms.get(room_code)

    def remove_student(self, socket_id):
        """Remove student by socket_id. Returns (room_code, student_name) or (None, None)."""
        with self.lock:
            room_code = self.socket_to_room.get(socket_id)
            if not room_code or room_code not in self.rooms:
                return None, None
            room = self.rooms[room_code]
            student_id = room.socket_to_student.get(socket_id)
            if not student_id or student_id not in room.students:
                return None, None
            name = room.students[student_id].name
            del room.students[student_id]
            del room.socket_to_student[socket_id]
            del self.socket_to_room[socket_id]
            return room_code, name

    def remove_teacher(self, socket_id):
        """Remove teacher by socket_id. Returns room_code or None."""
        with self.lock:
            for room_code, room in self.rooms.items():
                if room.teacher_socket == socket_id:
                    room.teacher_socket = None
                    return room_code
            return None

mgr = SessionManager()

# ── Helpers ────────────────────────────────────────────────────────────────────

def decode_image(b64_string):
    try:
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_data = base64.b64decode(b64_string)
        arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def extract_face(frame):
    """Detect face, crop, resize to 224x224, normalise to [0,1]. Returns None if no face."""
    if frame is None:
        return None
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = recognizer.face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    mx = int(w * recognizer.MARGIN_RATIO)
    my = int(h * recognizer.MARGIN_RATIO)
    x1 = max(0, x - mx);  y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (224, 224))
    return crop.astype(np.float32) / 255.0

def elapsed_str(room):
    if not room.start_time:
        return "00:00.000"
    e  = (datetime.now() - room.start_time).total_seconds()
    m  = int(e // 60)
    s  = int(e % 60)
    ms = int((e % 1) * 1000)
    return f"{m:02d}:{s:02d}.{ms:03d}"

def student_payload(student):
    """Serialisable dict for a student (no buffer)."""
    return {
        'id':         student.student_id,
        'name':       student.name,
        'connected':  student.connected,
        'join_time':  student.join_time,
        'emotion':    student.emotion,
        'confidence': student.confidence,
    }

def broadcast_student_list(room_code):
    room = mgr.get_room(room_code)
    if not room:
        return
    socketio.emit('student_list_update', {
        'students': [student_payload(s) for s in room.students.values()],
        'count':    len(room.students)
    }, room=room_code)

def aggregate_emotions(room):
    counts = defaultdict(int)
    total  = 0
    for s in room.students.values():
        counts[s.emotion] += 1
        total += 1
    if total == 0:
        return None
    pct = {e: counts[e] / total * 100 for e in counts}
    for e in ['Engagement', 'Boredom', 'Confusion', 'Frustration', 'Neutral']:
        pct.setdefault(e, 0.0)
        counts.setdefault(e, 0)

    alert = None
    if pct.get('Boredom', 0) > 50:
        alert = f"High boredom: {pct['Boredom']:.1f}% of students"
    elif pct.get('Confusion', 0) > 50:
        alert = f"High confusion: {pct['Confusion']:.1f}% of students"
    elif (pct.get('Boredom', 0) + pct.get('Confusion', 0) + pct.get('Frustration', 0)) > 70:
        alert = f"Low engagement: only {pct.get('Engagement', 0):.1f}% engaged"

    return {
        'timestamp':      elapsed_str(room),
        'total_students': total,
        'emotions':       dict(counts),
        'percentages':    pct,
        'alert':          alert,
    }

# ── HTTP Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/teacher/<room_code>')
def teacher_dashboard(room_code):
    if not mgr.get_room(room_code):
        return "Room not found", 404
    return render_template('teacher.html', room_code=room_code)

@app.route('/student/<room_code>')
def student_page(room_code):
    if not mgr.get_room(room_code):
        return "Room not found", 404
    return render_template('student.html', room_code=room_code)

@app.route('/review')
def review_page():
    return render_template('review.html')

@app.route('/api/create_room', methods=['POST'])
def create_room():
    data = request.json or {}
    teacher_name = data.get('teacher_name', '').strip()
    if not teacher_name:
        return jsonify({'success': False, 'error': 'Teacher name required'})
    code = mgr.create_room(teacher_name)
    print(f"Room created: {code} by {teacher_name}")
    return jsonify({'success': True, 'room_code': code})

@app.route('/api/verify_room', methods=['POST'])
def verify_room():
    data = request.json or {}
    code = data.get('room_code', '').strip()
    room = mgr.get_room(code)
    if room:
        return jsonify({'valid': True, 'teacher_name': room.teacher_name})
    return jsonify({'valid': False})

@app.route('/api/sessions')
def list_sessions():
    output_dir = 'classroom_sessions'
    if not os.path.exists(output_dir):
        return jsonify([])
    sessions = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            try:
                with open(os.path.join(output_dir, filename)) as f:
                    d = json.load(f)
                sessions.append({
                    'id':          d['session_id'],
                    'date':        d['session_id'][:8],
                    'duration':    d.get('duration', 'Unknown'),
                    'students':    d.get('total_students', 0),
                    'data_points': len(d.get('timeline', []))
                })
            except Exception:
                pass
    sessions.sort(key=lambda x: x['id'], reverse=True)
    return jsonify(sessions)

@app.route('/analytics/<session_id>')
def get_analytics(session_id):
    output_dir = 'classroom_sessions'
    if os.path.exists(output_dir):
        for fname in os.listdir(output_dir):
            if session_id in fname and fname.endswith('.json'):
                with open(os.path.join(output_dir, fname)) as f:
                    return jsonify(json.load(f))
    return jsonify({'error': 'Session not found'}), 404

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data = request.json or {}
    room_code = data.get('room_code')
    room = mgr.get_room(room_code)
    if not room:
        return jsonify({'success': False, 'error': 'Invalid room'})
    room.is_recording    = True
    room.start_time      = datetime.now()
    room.session_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    room.emotion_timeline = []
    print(f"Recording started in room {room_code}: {room.session_id}")
    socketio.emit('clear_alerts', room=room_code)
    socketio.emit('recording_started', {'session_id': room.session_id}, room=room_code)
    return jsonify({'success': True, 'session_id': room.session_id})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    data = request.json or {}
    room_code = data.get('room_code')
    room = mgr.get_room(room_code)
    if not room:
        return jsonify({'success': False, 'error': 'Invalid room'})
    if not room.is_recording:
        return jsonify({'success': False, 'error': 'Not recording'})
    room.is_recording = False
    os.makedirs('classroom_sessions', exist_ok=True)
    out_file = os.path.join('classroom_sessions',
                            f'room_{room_code}_session_{room.session_id}.json')
    report = {
        'room_code':       room_code,
        'session_id':      room.session_id,
        'teacher_name':    room.teacher_name,
        'start_time':      room.start_time.isoformat() if room.start_time else None,
        'duration':        elapsed_str(room),
        'total_students':  len(room.students),
        'students':        [student_payload(s) for s in room.students.values()],
        'timeline':        room.emotion_timeline,
        'total_data_points': len(room.emotion_timeline)
    }
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Recording stopped. Saved: {out_file} ({len(room.emotion_timeline)} points)")
    socketio.emit('recording_stopped', {
        'file': out_file, 'data_points': len(room.emotion_timeline)
    }, room=room_code)
    return jsonify({'success': True, 'file': out_file,
                    'data_points': len(room.emotion_timeline)})

# ── Socket handlers ────────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    print(f"Socket connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid

    # Check if teacher
    room_code = mgr.remove_teacher(sid)
    if room_code:
        print(f"Teacher disconnected from room {room_code}")
        socketio.emit('teacher_disconnected',
                      {'message': 'Teacher has ended the session'}, room=room_code)
        # Remove from socket_to_room if present
        mgr.socket_to_room.pop(sid, None)
        return

    # Check if student
    room_code, name = mgr.remove_student(sid)
    if room_code:
        print(f"Student disconnected from room {room_code}: {name}")
        broadcast_student_list(room_code)
    else:
        print(f"Unknown socket disconnected: {sid}")

@socketio.on('join_room')
def on_join_room(data):
    room_code  = data.get('room_code')
    is_teacher = data.get('is_teacher', False)
    room = mgr.get_room(room_code)
    if not room:
        return
    join_room(room_code)
    if is_teacher:
        room.teacher_socket = request.sid
        mgr.socket_to_room[request.sid] = room_code
        print(f"Teacher joined room {room_code} [{request.sid}]")
    else:
        print(f"Socket joined room {room_code} [{request.sid}]")

@socketio.on('student_join')
def on_student_join(data):
    sid        = request.sid
    student_id = data.get('student_id')
    name       = data.get('student_name', 'Unknown')
    room_code  = data.get('room_code')
    room = mgr.get_room(room_code)
    if not room:
        print(f"student_join: invalid room {room_code}")
        return

    # If this student_id already exists (reconnect), update socket
    if student_id in room.students:
        old_sid = room.students[student_id].socket_id
        room.socket_to_student.pop(old_sid, None)
        mgr.socket_to_room.pop(old_sid, None)

    student = Student(student_id, name, sid)
    room.students[student_id]      = student
    room.socket_to_student[sid]    = student_id
    mgr.socket_to_room[sid]        = room_code

    print(f"Student joined room {room_code}: {name} ({student_id}) [{sid}]")
    broadcast_student_list(room_code)

@socketio.on('student_leave')
def on_student_leave(data):
    sid        = request.sid
    student_id = data.get('student_id')
    room_code  = data.get('room_code')
    room = mgr.get_room(room_code)
    if not room:
        return
    if student_id in room.students:
        name   = room.students[student_id].name
        old_sid = room.students[student_id].socket_id
        del room.students[student_id]
        room.socket_to_student.pop(old_sid, None)
        mgr.socket_to_room.pop(old_sid, None)
        print(f"Student left room {room_code}: {name}")
        broadcast_student_list(room_code)
        # Immediately push fresh aggregate so teacher bars update
        room2 = mgr.get_room(room_code)
        if room2 is not None:
            if room2.students:
                agg = aggregate_emotions(room2)
                if agg:
                    socketio.emit('classroom_update', agg, room=room_code)
            else:
                socketio.emit('classroom_update', {
                    'timestamp': elapsed_str(room2),
                    'total_students': 0,
                    'emotions': {},
                    'percentages': {},
                    'alert': None,
                }, room=room_code)

@socketio.on('student_frame')
def on_student_frame(data):
    student_id = data.get('student_id')
    room_code  = data.get('room_code')
    frame_data = data.get('frame')
    room = mgr.get_room(room_code)
    if not room or student_id not in room.students:
        return

    frame = decode_image(frame_data)
    face  = extract_face(frame)

    student = room.students[student_id]

    if face is None:
        # No face — emit status but don't touch buffer
        socketio.emit('buffer_update', {
            'student_id':  student_id,
            'buffer_size': len(student.buffer),
            'buffer_full': len(student.buffer) >= 60,
            'face_found':  False
        }, room=room_code)
        return

    # Add to rolling buffer
    if len(student.buffer) >= 60:
        student.buffer.pop(0)
    student.buffer.append(face)

    socketio.emit('buffer_update', {
        'student_id':  student_id,
        'buffer_size': len(student.buffer),
        'buffer_full': len(student.buffer) >= 60,
        'face_found':  True
    }, room=room_code)

# ── Prediction scheduler ───────────────────────────────────────────────────────

def prediction_scheduler():
    print("Prediction scheduler started.")
    while True:
        time.sleep(2)
        try:
            for room_code, room in list(mgr.rooms.items()):
                if not room.students:
                    continue

                predicted_any = False

                for student_id, student in list(room.students.items()):
                    if len(student.buffer) < 60 or student.predicting:
                        continue

                    student.predicting = True
                    try:
                        preds = recognizer.predict_emotion(student.buffer)
                        if preds is None:
                            continue

                        emotion, confidence = recognizer.get_dominant_emotion(preds)

                        # Neutral if nothing crosses threshold
                        if emotion is None:
                            emotion    = "Neutral"
                            confidence = 0.0

                        student.emotion     = emotion
                        student.confidence  = float(confidence) if confidence else 0.0
                        student.all_scores  = {e: round(float(s), 3) for e, s in preds.items()}

                        print(f"[{room_code}] {student.name}: {emotion} "
                              f"({student.confidence:.2f}) | {student.all_scores}")

                        socketio.emit('emotion_update', {
                            'student_id': student_id,
                            'emotion':    emotion,
                            'confidence': student.confidence,
                            'all_scores': student.all_scores,
                            'status':     'detected'
                        }, room=room_code)

                        socketio.emit('student_emotion_update', {
                            'student_id': student_id,
                            'emotion':    emotion,
                            'confidence': student.confidence,
                        }, room=room_code)

                        predicted_any = True

                    except Exception as e:
                        print(f"Prediction error for {student.name}: {e}")
                        import traceback; traceback.print_exc()
                    finally:
                        student.predicting = False

                # Always broadcast aggregate after each prediction cycle,
                # even if only some students were predicted this round
                if room.students:
                    agg = aggregate_emotions(room)
                    if agg:
                        socketio.emit('classroom_update', agg, room=room_code)
                        if room.is_recording and predicted_any:
                            room.emotion_timeline.append(agg)
                            if agg.get('alert'):
                                socketio.emit('recording_alert', {
                                    'timestamp': agg['timestamp'],
                                    'alert':     agg['alert']
                                }, room=room_code)
                elif not room.students:
                    # Room now empty - push zeroed update
                    socketio.emit('classroom_update', {
                        'timestamp': elapsed_str(room),
                        'total_students': 0,
                        'emotions': {},
                        'percentages': {},
                        'alert': None,
                    }, room=room_code)

        except Exception as e:
            print(f"Scheduler error: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    t = threading.Thread(target=prediction_scheduler, daemon=True)
    t.start()

    print("\n" + "="*60)
    print("CLASSROOM EMOTION MONITORING SERVER")
    print("="*60)
    print("  Home:    http://localhost:5000")
    print("  Review:  http://localhost:5000/review")
    print("  Files needed: final_model_v2.h5, emotion_thresholds.json")
    print("="*60 + "\n")

    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
