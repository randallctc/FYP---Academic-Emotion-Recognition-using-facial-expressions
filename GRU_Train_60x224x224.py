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
MODEL_DIR = "/home/FYP/rand0019/FYP/ResNet50+GRU_Fast"
os.makedirs(MODEL_DIR, exist_ok=True)

with np.load(os.path.join(FEATURE_DIR, "train_features.npz")) as d:
    X_train = d["X"]
with np.load(os.path.join(FEATURE_DIR, "val_features.npz")) as d:
    X_val = d["X"]
with np.load(os.path.join(FEATURE_DIR, "test_features.npz")) as d:
    X_test = d["X"]

DATA_DIR = "/home/FYP/rand0019/FYP/Processed Data"
with np.load(os.path.join(DATA_DIR, "Train_60x224x224_every5th_25pct.npz")) as d:
    y_train = d["y"].astype(np.int32)
with np.load(os.path.join(DATA_DIR, "Validation_60x224x224_every5th_25pct.npz")) as d:
    y_val = d["y"].astype(np.int32)
with np.load(os.path.join(DATA_DIR, "Test_60x224x224_every5th_25pct.npz")) as d:
    y_test = d["y"].astype(np.int32)

def build_gru(input_shape, num_classes=4):
    model = models.Sequential([
        layers.GRU(256, return_sequences=False, input_shape=input_shape),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

affects = ["Boredom", "Engagement", "Confusion", "Frustration"]
results = {}
history_records = {}

for i, affect in enumerate(affects):
    print(f"\n=== Training GRU model for {affect} ===")

    y_train_s = y_train[:, i]
    y_val_s = y_val[:, i]
    y_test_s = y_test[:, i]

    model = build_gru(input_shape=X_train.shape[1:], num_classes=4)

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

    # Save training history (loss & accuracy)
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

    # Store for later aggregation
    history_records[affect] = history_dict

    # Clean up
    del best_model, model, history
    tf.keras.backend.clear_session()
    gc.collect()

results_path = os.path.join(MODEL_DIR, "final_results.csv")
pd.DataFrame(list(results.items()), columns=["Affect", "Test_Accuracy"]).to_csv(results_path, index=False)

# Also save all histories as .npz for quick reload
np.savez(os.path.join(MODEL_DIR, "all_histories.npz"), **history_records)

plt.bar(results.keys(), results.values(), color="cornflowerblue", edgecolor="black")
plt.ylabel("Accuracy")
plt.title("GRU (precomputed ResNet features) Accuracy per Affect")
plt.ylim(0, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.savefig(os.path.join(MODEL_DIR, "GRU_Accuracies.png"), dpi=300, bbox_inches="tight")
plt.show()

print("\n=== FINAL TEST RESULTS ===")
for affect, acc in results.items():
    print(f"{affect}: {acc:.4f}")
