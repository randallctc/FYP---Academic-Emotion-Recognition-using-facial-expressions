#!/bin/bash
# Runs before gunicorn — downloads all model files using plain Python
# before eventlet monkey-patches networking.

set -e

echo "[startup] Checking model files..."

python3 - << 'PYEOF'
import os, sys, requests

def download_hf(url, dest):
    if not url:
        print(f"[startup] No URL set for {dest}, skipping.")
        return
    if os.path.exists(dest):
        print(f"[startup] {dest} already present, skipping download.")
        return
    print(f"[startup] Downloading {dest} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"[startup] {dest} downloaded ({os.path.getsize(dest) // 1024 // 1024} MB).")

# ── GRU model + thresholds from Hugging Face ──────────────────────────────────
download_hf(os.environ.get("MODEL_URL", ""),  "final_model_v2.h5")
download_hf(os.environ.get("THRESH_URL", ""), "emotion_thresholds.json")

# ── ResNet50 ImageNet weights from Google ─────────────────────────────────────
# Keras caches weights in ~/.keras/models/. Download here before eventlet
# patches DNS, otherwise Keras can't resolve storage.googleapis.com.
KERAS_CACHE = os.path.expanduser("~/.keras/models")
RESNET_FILE = "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
RESNET_URL  = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
)
RESNET_PATH = os.path.join(KERAS_CACHE, RESNET_FILE)

os.makedirs(KERAS_CACHE, exist_ok=True)

if os.path.exists(RESNET_PATH):
    print(f"[startup] ResNet50 weights already cached, skipping download.")
else:
    print(f"[startup] Downloading ResNet50 weights (~90 MB)...")
    resp = requests.get(RESNET_URL, stream=True, timeout=300)
    resp.raise_for_status()
    with open(RESNET_PATH, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"[startup] ResNet50 weights downloaded ({os.path.getsize(RESNET_PATH) // 1024 // 1024} MB).")

print("[startup] All model files ready.")
PYEOF

echo "[startup] Starting gunicorn..."
exec gunicorn --worker-class eventlet -w 1 \
     --bind 0.0.0.0:5000 \
     --timeout 120 \
     run:flask_app
