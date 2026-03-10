import numpy as np
from tensorflow.keras import models, layers
import os

def build_lrcn(input_shape, num_classes=2):
    model = models.Sequential([
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),

        layers.TimeDistributed(layers.Flatten()),

        layers.LSTM(256, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

train_path = "/home/FYP/rand0019/FYP/Processed Data/Train_60x224x224_every5th_25pct.npz"
val_path = "/home/FYP/rand0019/FYP/Processed Data/Validation_60x224x224_every5th_25pct.npz"
test_path = "/home/FYP/rand0019/FYP/Processed Data/Test_60x224x224_every5th_25pct.npz"

train_data = np.load(train_path)
val_data = np.load(val_path)
test_data = np.load(test_path)

X_train, y_train = train_data["X"], train_data["y"]
X_val, y_val = val_data["X"], val_data["y"]
X_test, y_test = test_data["X"], test_data["y"]

print("Data loaded successfully")
print("Train shape:", X_train.shape, y_train.shape)
print("Validation shape:", X_val.shape, y_val.shape)
print("Test shape:", X_test.shape, y_test.shape)


# Step 2: Emotion Names
emotion_names = ["Boredom", "Engagement", "Confusion", "Frustration"]

# Create output folder for models
os.makedirs("saved_lrcn_binary_models", exist_ok=True)


# Step 3: Train one model per emotion
for idx, emotion in enumerate(emotion_names):
    print(f"Training model for: {emotion}")

    y_train_bin = (y_train[:, idx] >= 2).astype(np.int32)
    y_val_bin   = (y_val[:, idx] >= 2).astype(np.int32)
    y_test_bin  = (y_test[:, idx] >= 2).astype(np.int32)

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    model = build_lrcn(input_shape=input_shape, num_classes=2)

    # Train model
    history = model.fit(
        X_train, y_train_bin,
        validation_data=(X_val, y_val_bin),
        epochs=20,
        batch_size=8
    )

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test_bin)
    print(f"{emotion} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save model
    save_path = f"saved_lrcn_binary_models/LRCN_{emotion.lower()}.h5"
    model.save(save_path)
    print(f"Model saved at: {save_path}")
