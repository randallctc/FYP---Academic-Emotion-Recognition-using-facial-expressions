import os
import numpy as np
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import pandas as pd

# Directories
FEATURE_DIR = "/home/FYP/rand0019/FYP/ResNet_Features"
MODEL_DIR = "/home/FYP/rand0019/FYP/ResNet50+GRU"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load precomputed features
with np.load(os.path.join(FEATURE_DIR, "train_features.npz")) as d:
    X_train = d["X"]
with np.load(os.path.join(FEATURE_DIR, "val_features.npz")) as d:
    X_val = d["X"]
with np.load(os.path.join(FEATURE_DIR, "test_features.npz")) as d:
    X_test = d["X"]

# Load emotion labels (0-3 for each emotion)
DATA_DIR = "/home/FYP/rand0019/FYP/Processed Data"
with np.load(os.path.join(DATA_DIR, "Train_60x224x224_every5th_25pct.npz")) as d:
    y_train = d["y"].astype(int)
with np.load(os.path.join(DATA_DIR, "Validation_60x224x224_every5th_25pct.npz")) as d:
    y_val = d["y"].astype(int)
with np.load(os.path.join(DATA_DIR, "Test_60x224x224_every5th_25pct.npz")) as d:
    y_test = d["y"].astype(int)

# Emotion names
affects = ["Boredom", "Engagement", "Confusion", "Frustration"]

# Prepare labels for multi-head softmax (list of arrays)
y_train_heads = [y_train[:, i] for i in range(4)]
y_val_heads = [y_val[:, i] for i in range(4)]
y_test_heads = [y_test[:, i] for i in range(4)]

# Build GRU with 4 softmax heads (one per emotion, 4 levels: 0-3)
def build_gru_multihead(input_shape, num_emotions=4, num_levels=4):
    input_layer = Input(shape=input_shape)
    x = layers.GRU(256, return_sequences=False)(input_layer)
    x = layers.Dropout(0.5)(x)

    outputs = []
    for _ in range(num_emotions):
        head = layers.Dense(num_levels, activation='softmax')(x)
        outputs.append(head)

    model = models.Model(inputs=input_layer, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # integer labels 0-3
        metrics=['accuracy']
    )
    return model

# Build model
model = build_gru_multihead(input_shape=X_train.shape[1:])

# Callbacks
checkpoint_path = os.path.join(MODEL_DIR, "emotion_levels_best.h5")
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True,
                             monitor="val_loss", mode="min", verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5,
                           restore_best_weights=True, verbose=1)

# Training
history = model.fit(
    X_train, y_train_heads,
    validation_data=(X_val, y_val_heads),
    epochs=20,
    batch_size=8,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# Load best model
best_model = load_model(checkpoint_path)

# Evaluate
eval_results = best_model.evaluate(X_test, y_test_heads, verbose=0)
print("\nTest results per head (loss + accuracy per emotion):")
for i, affect in enumerate(affects):
    print(f"{affect}: Loss={eval_results[i]:.4f}, Accuracy={eval_results[i+4]:.4f}")

# Save final model
best_model.save(os.path.join(MODEL_DIR, "emotion_levels_final.h5"))

# Save training history
history_dict = {
    "epoch": np.arange(1, len(history.history["loss"]) + 1),
    "train_loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
    "train_accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
}
df_history = pd.DataFrame(history_dict)
csv_path = os.path.join(MODEL_DIR, "emotion_levels_history.csv")
df_history.to_csv(csv_path, index=False)
np.savez(os.path.join(MODEL_DIR, "emotion_levels_history.npz"), **history_dict)

# Plot average training & validation accuracy
plt.figure(figsize=(8,5))
plt.plot(history_dict["epoch"], history_dict["train_accuracy"], label="Train Accuracy")
plt.plot(history_dict["epoch"], history_dict["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("GRU Multi-Output Softmax: Emotion Levels (0-3)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "emotion_levels_training_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

# Clean up
del model, best_model, history
tf.keras.backend.clear_session()
gc.collect()
