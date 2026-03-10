import os
import numpy as np
from tensorflow.keras.applications import ResNet50

DATA_DIR = "/home/FYP/rand0019/FYP/Processed Data"
FEATURE_DIR = "/home/FYP/rand0019/FYP/ResNet_Features"
os.makedirs(FEATURE_DIR, exist_ok=True)

base_cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg")

def extract_features(X, name):
    num_samples, num_frames, h, w, c = X.shape
    X_flat = X.reshape((-1, h, w, c))
    feats_flat = base_cnn.predict(X_flat, batch_size=32, verbose=1)
    feats = feats_flat.reshape((num_samples, num_frames, -1))
    np.savez_compressed(os.path.join(FEATURE_DIR, f"{name}_features.npz"), X=feats)
    print(f"{name} features saved:", feats.shape)
    return feats

with np.load(os.path.join(DATA_DIR, "Train_60x224x224_every5th_25pct.npz")) as d:
    X_train = d["X"].astype(np.float32)
with np.load(os.path.join(DATA_DIR, "Validation_60x224x224_every5th_25pct.npz")) as d:
    X_val = d["X"].astype(np.float32)
with np.load(os.path.join(DATA_DIR, "Test_60x224x224_every5th_25pct.npz")) as d:
    X_test = d["X"].astype(np.float32)

extract_features(X_train, "train")
extract_features(X_val, "val")
extract_features(X_test, "test")
