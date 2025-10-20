import os
import numpy as np

# === CONFIG ===
DATA_DIR = r"/home/FYP/rand0019/FYP/Processed data"   # folder containing your .npz files
SUFFIX = "_float16"     # suffix for converted files

npz_files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".npz") and "float16" not in f.lower()
]
if not npz_files:
    print("No NPZ files found in the specified directory.")
    exit()

print(f"Found {len(npz_files)} NPZ files:")
for f in npz_files:
    print(" •", f)

print("\n=== Starting conversion to float16 ===\n")

for file_name in npz_files:
    src_path = os.path.join(DATA_DIR, file_name)
    dst_path = os.path.join(
        DATA_DIR,
        file_name.replace(".npz", f"{SUFFIX}.npz")
    )

    print(f"Processing {file_name} ...")

    data = np.load(src_path)
    new_data = {}

    for key in data.files:
        arr = data[key]
        old_dtype = arr.dtype

        # Only convert if it’s image/frame data (float or uint arrays)
        if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.uint8):
            arr = arr.astype(np.float16)
        new_data[key] = arr

        print(f"  {key}: {old_dtype} → {arr.dtype}, shape={arr.shape}")

    # Save new compressed float16 file
    np.savez_compressed(dst_path, **new_data)

    # Compare file sizes
    old_size = os.path.getsize(src_path) / (1024**2)
    new_size = os.path.getsize(dst_path) / (1024**2)
    reduction = 100 * (1 - new_size / old_size)

    print(f"Saved {os.path.basename(dst_path)}")
    print(f"Old size: {old_size:.2f} MB → New size: {new_size:.2f} MB ({reduction:.1f}% smaller)\n")

print("All conversions complete!")