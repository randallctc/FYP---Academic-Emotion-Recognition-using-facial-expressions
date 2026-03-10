import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Load dataset
data = np.load("daisee_dataset_max30.npz")
X, y = data["X"], data["y"]

print("X shape:", X.shape)  # (num_clips, frames, H, W, 3)
print("y shape:", y.shape)  # (num_clips, 4)

# Build LRCN model
def build_lrcn(input_shape, num_classes=4):
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'), input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),

        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),

        layers.TimeDistributed(layers.Flatten()),

        layers.LSTM(256, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer="adam", 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    return model

# Directory to save models
MODEL_DIR = "saved_lrcn_models"
os.makedirs(MODEL_DIR, exist_ok=True)

results = {}
affects = ["Boredom", "Engagement", "Confusion", "Frustration"]

for i, affect in enumerate(affects):
    print(f"\n=== Training model for {affect} ===")

    # Labels for this affect (ordinal: 0–3)
    y_affect = y[:, i]

    # Step 1: Train/Val/Test split (60/20/20)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_affect, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Step 2: Alternate-frame sampling
    X_train_sampled = X_train[:, ::2, :, :, :]
    X_val_sampled = X_val[:, ::2, :, :, :]
    X_test_sampled = X_test[:, ::2, :, :, :]

    # Step 3: Build model
    model = build_lrcn(input_shape=X_train_sampled.shape[1:], num_classes=4)

    # Save best checkpoint
    checkpoint_path = os.path.join(MODEL_DIR, f"{affect.lower()}_best.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max")

    # Train
    history = model.fit(
        X_train_sampled, y_train,
        validation_data=(X_val_sampled, y_val),
        epochs=20,
        batch_size=8,
        callbacks=[checkpoint],
        verbose=1
    )

    # Load best model
    best_model = models.load_model(checkpoint_path)

    # Step 4: Evaluate on test set
    test_loss, test_acc = best_model.evaluate(X_test_sampled, y_test, verbose=0)
    results[affect] = test_acc
    print(f"{affect} test accuracy: {test_acc:.4f}")

    # Step 5: Save final trained model (redundant but explicit)
    final_model_path = os.path.join(MODEL_DIR, f"{affect.lower()}_final.h5")
    best_model.save(final_model_path)
    print(f"Saved trained model for {affect} at {final_model_path}")

# Plot results
plt.bar(results.keys(), results.values())
plt.ylabel("Accuracy")
plt.title("LRCN Test Accuracy on DAiSEE (per affect)")
plt.ylim(0, 1.0)
plt.show()

print("Final Test Results:")
for affect, acc in results.items():
    print(f"{affect}: {acc:.4f}")
