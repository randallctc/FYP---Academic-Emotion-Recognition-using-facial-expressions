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
MODEL_DIR = "/home/FYP/rand0019/FYP/ResNet50+GRU_Fast_Binary_All"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load precomputed features
with np.load(os.path.join(FEATURE_DIR, "train_features.npz")) as d:
    X_train = d["X"]
with np.load(os.path.join(FEATURE_DIR, "val_features.npz")) as d:
    X_val = d["X"]
with np.load(os.path.join(FEATURE_DIR, "test_features.npz")) as d:
    X_test = d["X"]

# Load original labels
DATA_DIR = "/home/FYP/rand0019/FYP/Processed Data"
with np.load(os.path.join(DATA_DIR, "Train_60x224x224_every5th_25pct.npz")) as d:
    y_train = d["y"].astype(np.int32)
with np.load(os.path.join(DATA_DIR, "Validation_60x224x224_every5th_25pct.npz")) as d:
    y_val = d["y"].astype(np.int32)
with np.load(os.path.join(DATA_DIR, "Test_60x224x224_every5th_25pct.npz")) as d:
    y_test = d["y"].astype(np.int32)

# Binarize all emotions: 0-1 -> 0, 2-3 -> 1
y_train_bin = (y_train >= 2).astype(int)
y_val_bin = (y_val >= 2).astype(int)
y_test_bin = (y_test >= 2).astype(int)

# Emotion names
affects = ["Boredom", "Engagement", "Confusion", "Frustration"]

# Build GRU multi-output model (4 binary outputs)
def build_gru_multioutput_binary(input_shape, num_emotions=4):
    input_layer = Input(shape=input_shape)
    x = layers.GRU(256, return_sequences=False)(input_layer)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(num_emotions, activation="sigmoid")(x)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# Build model
model = build_gru_multioutput_binary(input_shape=X_train.shape[1:])

# Callbacks
checkpoint_path = os.path.join(MODEL_DIR, "all_emotions_best.h5")
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True,
                             monitor="val_loss", mode="min", verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=5,
                           restore_best_weights=True, verbose=1)

# Training
history = model.fit(
    X_train, y_train_bin,
    validation_data=(X_val, y_val_bin),
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# Load best model
best_model = load_model(checkpoint_path)

# Evaluate
test_loss, test_acc = best_model.evaluate(X_test, y_test_bin, verbose=0)
print("\nTest Loss:", test_loss)
print("Test Accuracy (averaged across 4 outputs):", test_acc)

# Save final model
best_model.save(os.path.join(MODEL_DIR, "all_emotions_final.h5"))

# Save training history
history_dict = {
    "epoch": np.arange(1, len(history.history["loss"]) + 1),
    "train_loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
    "train_accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
}
df_history = pd.DataFrame(history_dict)
csv_path = os.path.join(MODEL_DIR, "all_emotions_history.csv")
df_history.to_csv(csv_path, index=False)
np.savez(os.path.join(MODEL_DIR, "all_emotions_history.npz"), **history_dict)

# Plot training curve
plt.figure(figsize=(8,5))
plt.plot(history_dict["epoch"], history_dict["train_accuracy"], label="Train Accuracy")
plt.plot(history_dict["epoch"], history_dict["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("GRU Multi-Output Binary (All 4 Emotions)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(MODEL_DIR, "all_emotions_training_curve.png"), dpi=300, bbox_inches="tight")
plt.show()

# Clean up
del model, best_model, history
tf.keras.backend.clear_session()
gc.collect()
