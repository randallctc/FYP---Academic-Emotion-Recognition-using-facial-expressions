"""
config.py — All application constants in one place.
Change settings here; nothing else needs to be touched.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = "final_model_v2.h5"
THRESHOLD_PATH = "emotion_thresholds.json"
EMOTIONS       = ["Boredom", "Engagement", "Confusion", "Frustration"]

# ── Face detection ────────────────────────────────────────────────────────────
FRAME_SIZE   = 224          # pixels (square)
BUFFER_SIZE  = 60           # frames required before first prediction
MARGIN_RATIO = 0.40         # face crop margin as fraction of face size

# ── Prediction scheduler ──────────────────────────────────────────────────────
PREDICTION_INTERVAL = 2     # seconds between prediction cycles

# Seconds without a detected face before a student's emotion is cleared.
# Student shows "Away from camera" on teacher dashboard after this time.
ABSENT_SECONDS = 5

# Seconds to wait after face returns before running prediction again.
# Gives the rolling buffer time to fill with fresh frames.
REANALYZE_SECONDS = 2

# ── Alert thresholds (% of class) ────────────────────────────────────────────
ALERT_BOREDOM_PCT    = 50
ALERT_CONFUSION_PCT  = 50
ALERT_LOW_ENGAGE_PCT = 70   # combined boredom + confusion + frustration

# ── Flask / SocketIO ──────────────────────────────────────────────────────────
SECRET_KEY          = "classroom_secret_2024"
MAX_HTTP_BUFFER_MB  = 10
PING_TIMEOUT        = 120
PING_INTERVAL       = 25

# ── Session storage ───────────────────────────────────────────────────────────
SESSIONS_DIR = "classroom_sessions"
