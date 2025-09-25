from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import matplotlib.pyplot as plt

# ==== CONFIG ====
MODEL_DIR = "ResNet+GRU_per_affect_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# X and y should already be loaded
# X shape: (num_clips, frames, H, W, 3)
# y shape: (num_clips, 4)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ==== Build model function ====
def build_resnet_gru(input_shape, num_classes=4):
    base_cnn = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=input_shape[1:])
    for layer in base_cnn.layers:
        layer.trainable = False

    model = models.Sequential([
        layers.TimeDistributed(base_cnn, input_shape=input_shape),
        layers.TimeDistributed(layers.Flatten()),
        layers.GRU(256, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ==== Train per affect ====
affects = ["Boredom", "Engagement", "Confusion", "Frustration"]
results = {}

for i, affect in enumerate(affects):
    print(f"\n=== Training model for {affect} ===")

    # Labels for this affect (ordinal: 0–3)
    y_affect = y[:, i]

    # Train/Val/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_affect, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Alternate-frame sampling
    X_train_s = X_train[:, ::2, :, :, :]
    X_val_s = X_val[:, ::2, :, :, :]
    X_test_s = X_test[:, ::2, :, :, :]

    # Build model
    model = build_resnet_gru(input_shape=X_train_s.shape[1:], num_classes=4)

    # Checkpoint
    checkpoint_path = os.path.join(MODEL_DIR, f"{affect.lower()}_best.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max")

    # Train
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=20,
        batch_size=8,
        callbacks=[checkpoint],
        verbose=1
    )

    # Load best model
    best_model = models.load_model(checkpoint_path)

    # Evaluate
    test_loss, test_acc = best_model.evaluate(X_test_s, y_test, verbose=0)
    results[affect] = test_acc
    print(f"{affect} test accuracy: {test_acc:.4f}")

    # Save final model
    final_path = os.path.join(MODEL_DIR, f"{affect.lower()}_final.h5")
    best_model.save(final_path)
    print(f"Saved trained model for {affect} at {final_path}")

# ==== Plot per-affect results ====
plt.bar(results.keys(), results.values())
plt.ylabel("Accuracy")
plt.title("ResNet+GRU Test Accuracy per Affect")
plt.ylim(0, 1.0)
plt.show()

print("Final Test Results:")
for affect, acc in results.items():
    print(f"{affect}: {acc:.4f}")
