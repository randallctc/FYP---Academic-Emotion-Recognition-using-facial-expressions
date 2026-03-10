"""
app/__init__.py — Application factory.

Creates and wires together Flask, SocketIO, the ML predictor,
the session manager, and all route handlers.

Importing `socketio` from this module gives other modules access
to the shared SocketIO instance without circular imports.
"""

from flask import Flask
from flask_socketio import SocketIO

from app.config import (
    SECRET_KEY,
    MAX_HTTP_BUFFER_MB,
    PING_TIMEOUT,
    PING_INTERVAL,
    MODEL_PATH,
    THRESHOLD_PATH,
)
from app.models import SessionManager
from app.ml.predictor import EmotionPredictor

# ── Shared instances (importable by other modules) ────────────────────────────
flask_app = Flask(__name__, template_folder="../templates")
flask_app.config["SECRET_KEY"] = SECRET_KEY

socketio = SocketIO(
    flask_app,
    cors_allowed_origins="*",
    max_http_buffer_size=MAX_HTTP_BUFFER_MB * 1024 * 1024,
    ping_timeout=PING_TIMEOUT,
    ping_interval=PING_INTERVAL,
    async_mode="threading",
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
