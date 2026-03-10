"""
resave_local.py
===============
Extracts weights directly from the .keras file (which is just a zip),
loads them into a locally rebuilt model, then saves as .h5.

Run from Application_2.0 with AppEnv_tf213 activated:
    python resave_local.py
"""

import zipfile
import json
import numpy as np
import h5py
import os
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Bidirectional, GRU, Dense, Dropout, LayerNormalization
)

MODEL_KERAS = "final_model.keras"
MODEL_OUT   = "final_model_v2.h5"

# -- Step 1: Extract weights from .keras zip -----------------------------------
print("Extracting weights from .keras file...")
extract_dir = "_keras_extracted"
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(MODEL_KERAS, 'r') as z:
    z.extractall(extract_dir)
    print("Files inside .keras archive:")
    for f in z.namelist():
        print(" ", f)

# The weights are stored in model.weights.h5
weights_h5 = os.path.join(extract_dir, "model.weights.h5")
if not os.path.exists(weights_h5):
    weights_h5 = os.path.join(extract_dir, "variables", "variables.h5")

print("\nReading weights from: {}".format(weights_h5))

def print_h5_structure(name, obj):
    print(" ", name)

with h5py.File(weights_h5, 'r') as f:
    print("\nWeight file structure:")
    f.visititems(print_h5_structure)

# -- Step 2: Rebuild model architecture ----------------------------------------
print("\nBuilding model architecture...")

def build_model():
    inp = Input(shape=(60, 2048), name="sequence_input")
    x = LayerNormalization(name="layer_normalization")(inp)
    x = Bidirectional(
        GRU(128, return_sequences=True, dropout=0.5, recurrent_dropout=0.2),
        name="bigru_1"
    )(x)
    x = Bidirectional(
        GRU(64, return_sequences=False, dropout=0.5, recurrent_dropout=0.2),
        name="bigru_2"
    )(x)
    x = Dense(64, activation="relu", name="dense")(x)
    x = Dropout(0.5, name="dropout")(x)
    out_boredom     = Dense(1, activation="sigmoid", name="Boredom")(x)
    out_engagement  = Dense(1, activation="sigmoid", name="Engagement")(x)
    out_confusion   = Dense(1, activation="sigmoid", name="Confusion")(x)
    out_frustration = Dense(1, activation="sigmoid", name="Frustration")(x)
    return Model(
        inputs=inp,
        outputs={
            "Boredom":     out_boredom,
            "Engagement":  out_engagement,
            "Confusion":   out_confusion,
            "Frustration": out_frustration,
        },
        name="MultiHead_BiGRU"
    )

model = build_model()

# Build by running a dummy forward pass so all variables are created
dummy = np.zeros((1, 60, 2048), dtype=np.float32)
_ = model(dummy, training=False)
print("Model has {} weight tensors".format(len(model.weights)))

# -- Step 3: Map weights by layer name -----------------------------------------
print("\nLoading weights by layer name...")

with h5py.File(weights_h5, 'r') as f:
    all_keys = []
    f.visititems(lambda name, obj: all_keys.append(name) if isinstance(obj, h5py.Dataset) else None)

    loaded = 0
    for layer in model.layers:
        layer_weights = layer.weights
        if not layer_weights:
            continue

        layer_name = layer.name
        possible_paths = [
            "layers/{}/vars".format(layer_name),
            layer_name,
            "model/layers/{}/vars".format(layer_name),
        ]

        found_path = None
        for path in possible_paths:
            if path in f:
                found_path = path
                break

        if found_path is None:
            for key in all_keys:
                if layer_name in key and key.endswith("0"):
                    found_path = "/".join(key.split("/")[:-1])
                    break

        if found_path is None:
            print("  WARNING: Could not find weights for layer '{}'".format(layer_name))
            continue

        group = f[found_path]
        var_keys = sorted([k for k in group.keys()], key=lambda x: int(x))

        if len(var_keys) != len(layer_weights):
            print("  WARNING: Layer '{}' has {} weights but found {} in file".format(
                layer_name, len(layer_weights), len(var_keys)))
            continue

        weight_values = [group[k][()] for k in var_keys]
        layer.set_weights(weight_values)
        print("  Loaded {} weights for '{}'".format(len(weight_values), layer_name))
        loaded += 1

print("\nSuccessfully loaded weights for {} layers".format(loaded))

# -- Step 4: Save as .h5 -------------------------------------------------------
print("\nSaving to {}...".format(MODEL_OUT))
model.save(MODEL_OUT)
print("\nDone. Now update classroom_server.py to use '{}'".format(MODEL_OUT))

# Cleanup
import shutil
shutil.rmtree(extract_dir)