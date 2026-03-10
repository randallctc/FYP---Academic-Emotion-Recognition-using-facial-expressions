import pandas as pd

# Load original CSV
df = pd.read_csv("AllLabels.csv")

# Emotion columns
emotion_cols = ["Boredom", "Engagement", "Confusion", "Frustration"]

# Convert 0–1 → 0, 2–3 → 1
for col in emotion_cols:
    df[col] = (df[col] >= 2).astype(int)

# Save as new CSV
df.to_csv("AllLabels_binary.csv", index=False)

print("Done! Saved as AllLabels_binary.csv")
