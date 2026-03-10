# ── Base: slim Python 3.10 (last version TF 2.13 wheels are stable on) ────────
FROM python:3.10-slim

# ── System deps for OpenCV (headless) ─────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps first (cached layer — only re-runs when requirements change) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application package ────────────────────────────────────────────────────────
COPY run.py .
COPY app/ app/
COPY templates/ templates/

# Model files are NOT baked into the image.
# They are downloaded at container startup from Hugging Face
# using the MODEL_URL and THRESH_URL environment variables.

# ── Non-root user (Render security best practice) ─────────────────────────────
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

# Use gunicorn + eventlet worker for production SocketIO
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "run:flask_app"]
