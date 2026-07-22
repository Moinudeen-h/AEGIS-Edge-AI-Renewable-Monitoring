import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

base_dir = r"C:\Users\MOINODHEEN\Moinu\AEGIS_Project"
raw_log_path = os.path.join(base_dir, "data", "logs", "node1_log.csv")
clean_log_path = os.path.join(base_dir, "data", "logs", "node1_log_clean.csv")
arduino_dir = os.path.join(
    base_dir,
    "Arduino_AEGIS",
    "AEGIS_Node1_TFLite",
    "AEGIS_Node1_TFLite",
)
os.makedirs(arduino_dir, exist_ok=True)

print("=== AEGIS Node 1: Training local 3-feature autoencoder ===")
print(f"Reading raw log: {raw_log_path}")

# ------------------------------------------------------------------
# Step 1: clean raw serial log into a proper CSV
# Keep only lines that look like:
# timestamp_ms,temp_C,bus_V,current_mA_abs,power_mW
# i.e. exactly 4 commas (5 fields) and first char is a digit.
# ------------------------------------------------------------------
with open(raw_log_path, "r", encoding="utf-8", errors="ignore") as fin, \
     open(clean_log_path, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        if line.count(",") != 4:
            continue
        if not line[0].isdigit():
            # this also skips the header we printed from Arduino,
            # so we will add our own header next.
            continue
        fout.write(line + "\n")

print(f"Cleaned log written to: {clean_log_path}")

# Now read the clean log with a known header
col_names = ["timestamp_ms", "temp_C", "bus_V", "current_mA_abs", "power_mW"]
df = pd.read_csv(clean_log_path, header=None, names=col_names)
print(f"Rows in clean log: {len(df)}")

required_cols = ["temp_C", "bus_V", "current_mA_abs"]
X_raw = df[required_cols].astype(np.float32)
X_raw = X_raw.fillna(X_raw.mean())

print(f"Samples: {len(X_raw)}, features: {X_raw.shape[1]}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw.values)
print("StandardScaler fitted.")

input_dim = X_scaled.shape[1]
assert input_dim == 3, f"Expected 3 features, got {input_dim}"

X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Build tiny AE: 3 -> 4 -> 2 -> 4 -> 3
inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(4, activation="relu", name="enc1")(inputs)
x = layers.Dense(2, activation="relu", name="bottleneck")(x)
x = layers.Dense(4, activation="relu", name="dec1")(x)
outputs = layers.Dense(input_dim, activation="linear", name="out")(x)

ae = keras.Model(inputs, outputs, name="node1_local_ae")
ae.compile(optimizer="adam", loss="mse")

print(ae.summary())

history = ae.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    verbose=0,
)

print("Training done.")
print("Final train loss:", float(history.history["loss"][-1]))
print("Final val loss:", float(history.history["val_loss"][-1]))

# Reconstruction errors on all data
X_recon = ae.predict(X_scaled, verbose=0)
errors = np.mean(np.square(X_scaled - X_recon), axis=1)

print("Error stats on full log:")
print("  min:", float(errors.min()))
print("  max:", float(errors.max()))
print("  mean:", float(errors.mean()))
print("  std:", float(errors.std()))

threshold = np.percentile(errors, 95)
print("95th percentile threshold:", float(threshold))

results = {
    "threshold_95": float(threshold),
    "train_loss_final": float(history.history["loss"][-1]),
    "val_loss_final": float(history.history["val_loss"][-1]),
    "error_min": float(errors.min()),
    "error_max": float(errors.max()),
    "error_mean": float(errors.mean()),
    "error_std": float(errors.std()),
}
results_path = os.path.join(base_dir, "results", "node1_local_ae_results.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print("Saved results to", results_path)

# ---------------- TFLite INT8 conversion ----------------

def representative_dataset():
    for i in range(0, min(len(X_train), 500), 10):
        sample = X_train[i : i + 1].astype(np.float32)
        yield [sample]

converter = tf.lite.TFLiteConverter.from_keras_model(ae)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
print("TFLite INT8 model size (bytes):", len(tflite_model))

tflite_path = os.path.join(base_dir, "models", "tflite", "node1_local_ae_int8.tflite")
os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print("Saved TFLite model to", tflite_path)

# -------------- Write Arduino model header --------------

def bytes_to_c_array(var_name, data: bytes):
    hex_vals = ", ".join(f"0x{b:02x}" for b in data)
    return (
        f"// Auto-generated from node1_local_ae_int8.tflite\n"
        f"#ifndef NODE1_LOCAL_AE_INT8_H\n"
        f"#define NODE1_LOCAL_AE_INT8_H\n\n"
        f"const unsigned char {var_name}[] = {{\n"
        f"  {hex_vals}\n"
        f"}};\n\n"
        f"const unsigned int {var_name}_len = {len(data)};\n\n"
        f"#endif // NODE1_LOCAL_AE_INT8_H\n"
    )


model_header = bytes_to_c_array("node1_local_ae_int8_model", tflite_model)
model_header_path = os.path.join(arduino_dir, "node1_local_ae_int8.h")
with open(model_header_path, "w") as f:
    f.write(model_header)
print("Wrote model header to", model_header_path)

# ------------- Write Arduino scaler header --------------

means = scaler.mean_.astype(np.float32)
stds = scaler.scale_.astype(np.float32)

def format_array(name, values):
    body = ", ".join(f"{v:.8f}f" for v in values)
    return f"const float {name}[{len(values)}] = {{ {body} }};\n"

scaler_header = f"""// Auto-generated StandardScaler params for Node 1 local AE

#ifndef AEGIS_NODE1_SCALER_H
#define AEGIS_NODE1_SCALER_H

{format_array("NODE1_FEATURE_MEAN", means)}
{format_array("NODE1_FEATURE_STD", stds)}

#endif // AEGIS_NODE1_SCALER_H
"""

scaler_header_path = os.path.join(arduino_dir, "aegis_node1_scaler.h")
with open(scaler_header_path, "w") as f:
    f.write(scaler_header)
print("Wrote scaler header to", scaler_header_path)

print("\n=== Done. Copy this threshold into Arduino code: ===")
print("NODE1_LOCAL_AE_THRESHOLD =", float(threshold))
