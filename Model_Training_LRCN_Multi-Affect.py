import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

data = np.load("daisee_dataset_max30.npz")
X, y = data["X"], data["y"]

print("X shape:", X.shape)  # (num_clips, frames, H, W, 3)
print("y shape:", y.shape)  # (num_clips, 4)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = X_train[:, ::2, :, :, :]
X_val = X_val[:, ::2, :, :, :]
X_test = X_test[:, ::2, :, :, :]

input_layer = Input(shape=X_train.shape[1:])  # (frames, H, W, 3)

x = layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'))(input_layer)
x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)
x = layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu'))(x)
x = layers.TimeDistributed(layers.MaxPooling2D((2,2)))(x)
x = layers.TimeDistributed(layers.Flatten())(x)

x = layers.LSTM(256)(x)
x = layers.Dropout(0.5)(x)

output_boredom = layers.Dense(4, activation='softmax', name='Boredom')(x)
output_engagement = layers.Dense(4, activation='softmax', name='Engagement')(x)
output_confusion = layers.Dense(4, activation='softmax', name='Confusion')(x)
output_frustration = layers.Dense(4, activation='softmax', name='Frustration')(x)

model = Model(inputs=input_layer, outputs=[output_boredom, output_engagement, output_confusion, output_frustration])

model.compile(
    optimizer='adam',
    loss={
        'Boredom': 'sparse_categorical_crossentropy',
        'Engagement': 'sparse_categorical_crossentropy',
        'Confusion': 'sparse_categorical_crossentropy',
        'Frustration': 'sparse_categorical_crossentropy'
    },
    metrics={
        'Boredom': ['accuracy'],
        'Engagement': ['accuracy'],
        'Confusion': ['accuracy'],
        'Frustration': ['accuracy']
    }
)

model.summary()

y_train_dict = {
    "Boredom": y_train[:, 0],
    "Engagement": y_train[:, 1],
    "Confusion": y_train[:, 2],
    "Frustration": y_train[:, 3]
}

y_val_dict = {
    "Boredom": y_val[:, 0],
    "Engagement": y_val[:, 1],
    "Confusion": y_val[:, 2],
    "Frustration": y_val[:, 3]
}

y_test_dict = {
    "Boredom": y_test[:, 0],
    "Engagement": y_test[:, 1],
    "Confusion": y_test[:, 2],
    "Frustration": y_test[:, 3]
}

MODEL_DIR = "saved_lrcn_models"
os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, "lrcn_multi_best_2.h5")
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", mode="min")

history = model.fit(
    X_train, y_train_dict,
    validation_data=(X_val, y_val_dict),
    epochs=20,
    batch_size=8,
    callbacks=[checkpoint],
    verbose=1
)

model.load_weights(checkpoint_path)

# ✅ Evaluate and extract individual metrics
results = model.evaluate(X_test, y_test_dict, verbose=1, batch_size=8)

print("\n" + "="*60)
print("TEST RESULTS - INDIVIDUAL METRICS")
print("="*60)

# Extract metrics properly
metric_names = model.metrics_names
metric_values = results

# Create a clean display
for name, value in zip(metric_names, metric_values):
    print(f"{name:<30}: {value:.4f}")

print("="*60)

print("\nACCURACY SUMMARY:")
print("-" * 40)
for affect in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
    # Find the accuracy metric for this affect
    acc_key = f'{affect}_accuracy'
    if acc_key in metric_names:
        idx = metric_names.index(acc_key)
        accuracy = metric_values[idx] * 100
        print(f"{affect:<15}: {accuracy:>6.2f}%")
print("-" * 40)

print("\nLOSS SUMMARY:")
print("-" * 40)
for affect in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
    # Find the loss metric for this affect
    loss_key = f'{affect}_loss'
    if loss_key in metric_names:
        idx = metric_names.index(loss_key)
        loss_value = metric_values[idx]
        print(f"{affect:<15}: {loss_value:>6.4f}")
print("-" * 40)

# Save final model
final_model_path = os.path.join(MODEL_DIR, "lrcn_multi_final.h5")
model.save(final_model_path)
print(f"\nSaved final multi-affect LRCN model at {final_model_path}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors = {'Boredom': '#e74c3c', 'Engagement': '#3498db', 
          'Confusion': '#f39c12', 'Frustration': '#9b59b6'}

for affect in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
    ax1.plot(history.history[f'{affect}_accuracy'], 
             label=f'{affect} (Train)', color=colors[affect], linewidth=2)
    ax1.plot(history.history[f'val_{affect}_accuracy'], 
             '--', label=f'{affect} (Val)', color=colors[affect], linewidth=2, alpha=0.7)

ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1])

# Plot 2: Training and Validation Loss
for affect in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
    ax2.plot(history.history[f'{affect}_loss'], 
             label=f'{affect} (Train)', color=colors[affect], linewidth=2)
    ax2.plot(history.history[f'val_{affect}_loss'], 
             '--', label=f'{affect} (Val)', color=colors[affect], linewidth=2, alpha=0.7)

ax2.set_xlabel('Epochs', fontsize=12, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nTraining history plot saved as 'training_history.png'")

# ✅ Create final accuracy bar chart
final_accuracies = {}
for affect in ['Boredom', 'Engagement', 'Confusion', 'Frustration']:
    acc_key = f'{affect}_accuracy'
    if acc_key in metric_names:
        idx = metric_names.index(acc_key)
        final_accuracies[affect] = metric_values[idx] * 100

plt.figure(figsize=(10, 6))
bars = plt.bar(final_accuracies.keys(), final_accuracies.values(), 
               color=['#e74c3c', '#3498db', '#f39c12', '#9b59b6'],
               edgecolor='black', linewidth=1.5)

plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.xlabel('Emotions', fontsize=12, fontweight='bold')
plt.title('Final Test Accuracy by Emotion', fontsize=14, fontweight='bold')
plt.ylim(0, 100)
plt.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
plt.grid(axis='y', alpha=0.3)
plt.legend()

# Add percentage labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('final_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

print("Final accuracy chart saved as 'final_accuracy.png'")