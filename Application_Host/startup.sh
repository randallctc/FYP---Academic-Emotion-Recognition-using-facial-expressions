#!/bin/bash
# Runs before gunicorn starts — downloads model files using standard Python
# (before eventlet monkey-patches networking).

set -e

echo "[startup] Checking model files..."

python3 - << 'PYEOF'
import os, requests, sys

def download(url, dest):
    if not url:
        print(f"[startup] No URL set for {dest}, skipping.")
        return
    if os.path.exists(dest):
        print(f"[startup] {dest} already present, skipping download.")
        return
    print(f"[startup] Downloading {dest} ...")
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        mb = os.path.getsize(dest) // 1024 // 1024
        print(f"[startup] {dest} downloaded ({mb} MB).")
    except Exception as e:
        print(f"[startup] ERROR downloading {dest}: {e}", file=sys.stderr)
        sys.exit(1)

download(os.environ.get("MODEL_URL", ""),  "final_model_v2.h5")
download(os.environ.get("THRESH_URL", ""), "emotion_thresholds.json")
print("[startup] Model files ready.")
PYEOF

echo "[startup] Starting gunicorn..."
exec gunicorn --worker-class eventlet -w 1 \
     --bind 0.0.0.0:5000 \
     --timeout 120 \
     run:flask_app
