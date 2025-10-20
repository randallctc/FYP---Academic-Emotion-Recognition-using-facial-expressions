import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
import pandas as pd

FEATURE_DIR = "/home/FYP/rand0019/FYP/ResNet_Features"
MODEL_DIR = "/home/FYP/rand0019/FYP/ResNet50+GRU_Fast_Binary"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load features
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

# GRU model for binary classification
def build_gru_binary(input_shape):
    model = models.Sequential([
        layers.GRU(256, return_sequences=False, input_shape=input_shape),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# Emotion names
affects = ["Boredom", "Engagement", "Confusion", "Frustration"]
results = {}
history_records = {}

# Train one model per emotion
for i, affect in enumerate(affects):
    print(f"\n=== Training GRU binary model for {affect} ===")

    y_train_s = y_train_bin[:, i]
    y_val_s = y_val_bin[:, i]
    y_test_s = y_test_bin[:, i]

    model = build_gru_binary(input_shape=X_train.shape[1:])

    checkpoint_path = os.path.join(MODEL_DIR, f"{affect.lower()}_best.h5")
    
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True,
                                 monitor="val_accuracy", mode="max", verbose=1)
    
    early_stop = EarlyStopping(monitor="val_accuracy", patience=5,
                               restore_best_weights=True, verbose=1)

    history = model.fit(
        X_train, y_train_s,
        validation_data=(X_val, y_val_s),
        epochs=30,
        batch_size=32,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    best_model = load_model(checkpoint_path)
    test_loss, test_acc = best_model.evaluate(X_test, y_test_s, verbose=0)
    results[affect] = test_acc
    print(f"{affect} Test Accuracy: {test_acc:.4f}")

    best_model.save(os.path.join(MODEL_DIR, f"{affect.lower()}_final.h5"))

    # Save training history
    history_dict = {
        "epoch": np.arange(1, len(history.history["loss"]) + 1),
        "train_loss": history.history["loss"],
        "train_accuracy": history.history["accuracy"],
        "val_loss": history.history["val_loss"],
        "val_accuracy": history.history["val_accuracy"]
    }
    df_history = pd.DataFrame(history_dict)
    csv_path = os.path.join(MODEL_DIR, f"{affect.lower()}_history.csv")
    df_history.to_csv(csv_path, index=False)

    history_records[affect] = history_dict

    # Clean up
    del best_model, model, history
    tf.keras.backend.clear_session()
    gc.collect()

# Compute per-emotion accuracy and store
results = {}
for i, affect in enumerate(affects):
    correct = np.sum(pred_labels[:, i] == y_test_bin[:, i])
    acc = correct / len(y_test_bin)
    results[affect] = acc

# Save final results
results_path = os.path.join(MODEL_DIR, "final_results.csv")
pd.DataFrame(list(results.items()), columns=["Affect", "Test_Accuracy"]).to_csv(results_path, index=False)

# Save all histories as .npz
history_records = {
    "epoch": np.arange(1, len(history.history["loss"]) + 1),
    "train_loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
    "train_accuracy": history.history["accuracy"],
    "val_accuracy": history.history["val_accuracy"]
}
np.savez(os.path.join(MODEL_DIR, "all_histories.npz"), **history_records)

# Plot test accuracies
plt.bar(results.keys(), results.values(), color="cornflowerblue", edgecolor="black")
plt.ylabel("Accuracy")
plt.title("GRU (precomputed ResNet features) Binary Accuracy per Affect")
plt.ylim(0, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(os.path.join(MODEL_DIR, "GRU_Binary_Accuracies.png"), dpi=300, bbox_inches="tight")
plt.show()

# Print final test accuracies
print("\n=== FINAL TEST RESULTS ===")
for affect, acc in results.items():
    print(f"{affect}: {acc:.4f}")

