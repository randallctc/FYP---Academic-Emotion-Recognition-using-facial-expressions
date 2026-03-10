"""
run.py — Server entry point.

Run with:
    python run.py
"""

from app import flask_app, socketio
from app.config import MODEL_PATH, THRESHOLD_PATH

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CLASSROOM EMOTION MONITORING SERVER")
    print("=" * 60)
    print("  Home:    http://localhost:5000")
    print("  Review:  http://localhost:5000/review")
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Thresholds: {THRESHOLD_PATH}")
    print("=" * 60 + "\n")

    socketio.run(flask_app, host="0.0.0.0", port=5000, debug=False)
