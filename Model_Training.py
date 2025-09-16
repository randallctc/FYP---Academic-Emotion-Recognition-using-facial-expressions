import numpy as np
from sklearn.model_selection import train_test_split
import os

from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed dataset
data = np.load("daisee_dataset.npz", allow_pickle=True)
X, y = data["X"], data["y"]

print("X shape:", X.shape)  # (num_samples, max_frames, 64, 64, 3)
print("y shape:", y.shape)  # (num_samples, 4)

# Normalize pixel values (0–255 → 0–1)
X = X.astype("float32") / 255.0

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras import layers, models

max_frames = X.shape[1]   # e.g. 30
img_height, img_width = X.shape[2:4]
num_classes = y.shape[1]  # 4 (Boredom, Engagement, Confusion, Frustration)

def build_lrcn():
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'), input_shape=(max_frames, img_height, img_width, 3)),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        
        layers.TimeDistributed(layers.Flatten()),
        
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='sigmoid')  # multi-label outputs
    ])
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = build_lrcn()
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,          # try more later
    batch_size=8        # adjust based on GPU memory
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f}")

import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="train acc")
plt.plot(history.history["val_accuracy"], label="val acc")
plt.legend()
plt.show()
