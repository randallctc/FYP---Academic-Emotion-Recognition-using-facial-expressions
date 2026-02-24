"""
Simple Classroom Emotion Monitoring Server
Run this on teacher's computer, students connect via browser
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from datetime import datetime
import json
import os
from collections import defaultdict
from emotion_recognition_app import AcademicEmotionRecognizer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'classroom_secret'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10*1024*1024)

# Initialize emotion recognizer
print("Loading emotion recognition model...")
recognizer = AcademicEmotionRecognizer("all_emotions_final.h5")
print("Model loaded successfully!")

# Session data
class ClassroomSession:
    def __init__(self):
        self.students = {}  # {student_id: {name, buffer, current_emotion}}
        self.emotion_timeline = []  # Aggregated timeline
        self.is_recording = False
        self.start_time = None
        self.session_id = None
    
    def reset(self):
        self.students = {}
        self.emotion_timeline = []
        self.is_recording = False
        self.start_time = None
        self.session_id = None

session = ClassroomSession()

def decode_base64_image(base64_string):
    """Decode base64 image to numpy array"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def process_face(frame):
    """Detect and crop face from frame"""
    if frame is None:
        return None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = recognizer.face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )
    
    if len(faces) == 0:
        return None
    
    # Get first face
    x, y, w, h = faces[0]
    mx = int(w * recognizer.MARGIN_RATIO)
    my = int(h * recognizer.MARGIN_RATIO)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)
    
    face_crop = frame[y1:y2, x1:x2]
    
    if face_crop.size == 0:
        return None
    
    # Resize and normalize
    face_crop = cv2.resize(face_crop, (recognizer.FRAME_SIZE, recognizer.FRAME_SIZE))
    face_crop = face_crop.astype(np.float32) / 255.0
    
    return face_crop

def get_elapsed_time():
    """Get elapsed time since recording started"""
    if not session.start_time:
        return "00:00.000"
    
    elapsed = (datetime.now() - session.start_time).total_seconds()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    milliseconds = int((elapsed % 1) * 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

def aggregate_emotions():
    """Aggregate emotions from all students"""
    emotion_counts = defaultdict(int)
    total_students = 0
    
    for student_id, student_data in session.students.items():
        if 'current_emotion' in student_data:
            emotion = student_data['current_emotion']['emotion']
            emotion_counts[emotion] += 1
            total_students += 1
    
    if total_students == 0:
        return None
    
    # Calculate percentages
    percentages = {
        emotion: (count / total_students * 100)
        for emotion, count in emotion_counts.items()
    }
    
    # Check for alerts
    alert = None
    if percentages.get('Boredom', 0) > 50:
        alert = f"⚠️ High boredom: {percentages['Boredom']:.1f}% of students"
    elif percentages.get('Confusion', 0) > 50:
        alert = f"⚠️ High confusion: {percentages['Confusion']:.1f}% of students"
    elif (percentages.get('Boredom', 0) + percentages.get('Confusion', 0) + 
          percentages.get('Frustration', 0)) > 70:
        alert = f"⚠️ Low engagement: Only {percentages.get('Engagement', 0):.1f}% engaged"
    
    aggregated = {
        'timestamp': get_elapsed_time(),
        'total_students': total_students,
        'emotions': dict(emotion_counts),
        'percentages': percentages,
        'alert': alert
    }
    
    return aggregated

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/teacher')
def teacher_dashboard():
    return render_template('teacher.html')

@app.route('/student')
def student_page():
    return render_template('student.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    session.is_recording = True
    session.start_time = datetime.now()
    session.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session.emotion_timeline = []
    
    print(f"Recording started: {session.session_id}")
    socketio.emit('recording_started', {'session_id': session.session_id})
    
    return jsonify({'success': True, 'session_id': session.session_id})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    if not session.is_recording:
        return jsonify({'success': False, 'error': 'Not recording'})
    
    session.is_recording = False
    
    # Save results
    output_dir = 'classroom_sessions'
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f'session_{session.session_id}.json')
    
    report = {
        'session_id': session.session_id,
        'start_time': session.start_time.isoformat(),
        'duration': get_elapsed_time(),
        'total_students': len(session.students),
        'students': list(session.students.values()),
        'timeline': session.emotion_timeline
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Recording stopped. Saved to: {output_file}")
    socketio.emit('recording_stopped', {'file': output_file})
    
    return jsonify({'success': True, 'file': output_file})

@app.route('/analytics/<session_id>')
def get_analytics(session_id):
    output_file = os.path.join('classroom_sessions', f'session_{session_id}.json')
    
    if not os.path.exists(output_file):
        return jsonify({'error': 'Session not found'}), 404
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    return jsonify(data)

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    print(f"Client connected")

@socketio.on('student_join')
def handle_student_join(data):
    student_id = data['student_id']
    student_name = data['student_name']
    
    session.students[student_id] = {
        'id': student_id,
        'name': student_name,
        'buffer': [],
        'connected': True,
        'join_time': datetime.now().isoformat()
    }
    
    print(f"Student joined: {student_name} ({student_id})")
    
    # Notify teacher
    socketio.emit('student_list_update', {
        'students': list(session.students.values()),
        'count': len(session.students)
    })

@socketio.on('student_frame')
def handle_student_frame(data):
    student_id = data['student_id']
    frame_data = data['frame']
    
    if student_id not in session.students:
        return
    
    # Decode frame
    frame = decode_base64_image(frame_data)
    if frame is None:
        return
    
    # Process face
    face_crop = process_face(frame)
    if face_crop is None:
        return
    
    # Add to buffer
    session.students[student_id]['buffer'].append(face_crop)
    
    # Keep only last 60 frames
    if len(session.students[student_id]['buffer']) > 60:
        session.students[student_id]['buffer'].pop(0)
    
    # Predict if buffer is full
    if len(session.students[student_id]['buffer']) == 60:
        try:
            predictions = recognizer.predict_emotion(session.students[student_id]['buffer'])
            
            if predictions is not None:
                emotion, confidence = recognizer.get_dominant_emotion(predictions)
                
                session.students[student_id]['current_emotion'] = {
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'all_scores': {
                        recognizer.emotions[i]: float(predictions[i])
                        for i in range(len(recognizer.emotions))
                    }
                }
                
                # Aggregate and send to teacher
                if session.is_recording:
                    aggregated = aggregate_emotions()
                    if aggregated:
                        session.emotion_timeline.append(aggregated)
                        socketio.emit('classroom_update', aggregated)
        
        except Exception as e:
            print(f"Error processing emotions: {e}")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CLASSROOM EMOTION MONITORING SERVER")
    print("="*60)
    print("\nServer starting on http://localhost:5000")
    print("\nURLs:")
    print("  Teacher Dashboard: http://localhost:5000/teacher")
    print("  Student Page:      http://localhost:5000/student")
    print("\nMake sure 'all_emotions_final.h5' is in the current directory!")
    print("="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
