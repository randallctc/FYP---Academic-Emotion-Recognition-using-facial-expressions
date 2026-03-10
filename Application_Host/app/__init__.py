"""
app/__init__.py — Application factory.

Creates and wires together Flask, SocketIO, the ML predictor,
the session manager, and all route handlers.

Importing `socketio` from this module gives other modules access
to the shared SocketIO instance without circular imports.
"""

from flask import Flask
from flask_socketio import SocketIO
import os
import requests

from app.config import (
    SECRET_KEY,
    MAX_HTTP_BUFFER_MB,
    PING_TIMEOUT,
    PING_INTERVAL,
    MODEL_PATH,
    THRESHOLD_PATH,
    MODEL_URL,
    THRESHOLD_URL,
)
from app.models import SessionManager
from app.ml.predictor import EmotionPredictor

# ── Download model files from Hugging Face if not present locally ─────────────
def _download(url: str, dest: str) -> None:
    if not url:
        return
    if os.path.exists(dest):
        print(f"[startup] {dest} already present, skipping download.")
        return
    print(f"[startup] Downloading {dest} from {url} ...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[startup] {dest} downloaded ({os.path.getsize(dest) // 1024 // 1024} MB).")

_download(MODEL_URL, MODEL_PATH)
_download(THRESHOLD_URL, THRESHOLD_PATH)

# ── Shared instances (importable by other modules) ────────────────────────────
flask_app = Flask(__name__, template_folder="../templates")
flask_app.config["SECRET_KEY"] = SECRET_KEY

socketio = SocketIO(
    flask_app,
    cors_allowed_origins="*",
    max_http_buffer_size=MAX_HTTP_BUFFER_MB * 1024 * 1024,
    ping_timeout=PING_TIMEOUT,
    ping_interval=PING_INTERVAL,
    async_mode="eventlet",
)

mgr       = SessionManager()
predictor = EmotionPredictor(MODEL_PATH, THRESHOLD_PATH)

# ── Register routes and socket handlers ───────────────────────────────────────
from app.routes.http     import http_bp, init_routes
from app.routes.sockets  import init_handlers
from app.services.scheduler import start_scheduler

init_routes(mgr)
init_handlers(mgr, socketio)
flask_app.register_blueprint(http_bp)

# ── Start background prediction loop ─────────────────────────────────────────
start_scheduler(mgr, predictor, socketio)
