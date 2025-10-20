import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import random

BASE_DIR = r"/home/FYP/rand0019/FYP/DAiSEE/Data"
LABEL_FILE = r"/home/FYP/rand0019/FYP/DAiSEE/LAbels/AllLabels.csv"
SAVE_DIR = r"/home/FYP/rand0019/FYP/Processed Data"

os.makedirs(SAVE_DIR, exist_ok=True)

SETS = ["Train", "Validation", "Test"]
SAMPLE_FRACTION = 0.25
FRAME_STEP = 5
MAX_FRAMES = 60
IMG_SIZE = (224, 224)
MIN_FRAMES = 5

print("Loading labels...")
labels_df = pd.read_csv(LABEL_FILE)
labels_df.columns = labels_df.columns.str.strip()

labels_dict = {
    row["ClipID"].replace(".avi", ""): np.array([
        row["Boredom"], row["Engagement"], row["Confusion"], row["Frustration"]
    ], dtype=np.float32)
    for _, row in labels_df.iterrows()
}

print(f"Loaded {len(labels_dict)} label entries")

# =========================
# FRAME EXTRACTION FUNCTION
# =========================
def get_frames_for_clip(clip_path, max_frames=MAX_FRAMES, img_size=IMG_SIZE,
                        min_frames=MIN_FRAMES, frame_step=FRAME_STEP):
    frame_files = sorted([f for f in os.listdir(clip_path)
                          if f.lower().endswith(".jpg")])[::frame_step]
    frames = []

    for fname in frame_files:
        img = cv2.imread(os.path.join(clip_path, fname))
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        frames.append(img)

    if len(frames) < min_frames:
        return None

    # Pad or truncate to max_frames
    if len(frames) > max_frames:
        frames = frames[:max_frames]
    elif len(frames) < max_frames:
        pad = np.zeros((max_frames - len(frames), *img_size, 3), dtype=np.uint8)
        frames.extend(list(pad))

    return np.array(frames, dtype=np.float16) / 255.0


def process_dataset(split_name):
    print(f"\nProcessing split: {split_name}")
    split_dir = os.path.join(BASE_DIR, split_name)

    # Get valid clip IDs
    all_clip_ids = []
    for clip_id, lbl in labels_dict.items():
        prefix_folder = clip_id[:6]
        clip_path = os.path.join(split_dir, prefix_folder, clip_id)
        if os.path.exists(clip_path) and os.path.isdir(clip_path):
            all_clip_ids.append(clip_id)

    print(f"Found {len(all_clip_ids)} clips in {split_name}")

    num_samples = max(1, int(len(all_clip_ids) * SAMPLE_FRACTION))
    sampled_ids = random.sample(all_clip_ids, num_samples)
    print(f"Using {len(sampled_ids)} clips ({SAMPLE_FRACTION*100:.0f}% sample)")

    X, y = [], []
    num_skipped = 0

    for clip_id in tqdm(sampled_ids, desc=f"Extracting {split_name} frames"):
        prefix_folder = clip_id[:6]
        clip_path = os.path.join(split_dir, prefix_folder, clip_id)
        frames = get_frames_for_clip(clip_path)

        if frames is not None:
            X.append(frames)
            y.append(labels_dict[clip_id])
        else:
            num_skipped += 1

        if frames is not None:
            del frames
        gc.collect()

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)
    print(f"{split_name}: {X.shape[0]} samples, {num_skipped} skipped")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    save_path = os.path.join(SAVE_DIR, f"{split_name}_60x224x224_every5th_25pct.npz")
    np.savez_compressed(save_path, X=X, y=y)

    print(f"💾 Saved {split_name} dataset to {save_path}")
    return save_path, X, y

print("\nAll done!")

