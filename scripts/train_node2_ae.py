import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = r"C:\Users\MOINODHEEN\Moinu\AEGIS_Project"
RAW_CSV     = os.path.join(BASE, "data", "logs",   "node2_log.csv")
CLEAN_CSV   = os.path.join(BASE, "data", "logs",   "node2_log_clean.csv")
RESULTS_JSON= os.path.join(BASE, "results",         "node2_local_ae_results.json")
TFLITE_PATH = os.path.join(BASE, "models", "tflite","node2_local_ae_int8.tflite")
MODEL_H     = os.path.join(BASE, "Arduino_AEGIS", "AEGIS_Node2_TFLite",
                           "AEGIS_Node2_TFLite",   "node2_local_ae_int8.h")
SCALER_H    = os.path.join(BASE, "Arduino_AEGIS", "AEGIS_Node2_TFLite",
                           "AEGIS_Node2_TFLite",   "aegis_node2_scaler.h")

os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
os.makedirs(os.path.dirname(TFLITE_PATH),  exist_ok=True)
os.makedirs(os.path.dirname(MODEL_H),      exist_ok=True)

# ── Features (4 features for Node 2) ──────────────────────────────────────────
FEATURES    = ["bus_V", "current_mA_abs", "power_mW", "lux"]
NUM_FEATURES= len(FEATURES)

print("=== AEGIS Node 2: Training local 4-feature autoencoder ===")
print(f"Reading raw log: {RAW_CSV}")

# ── Load & clean ───────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)
df.columns = df.columns.str.strip()

# Drop rows with any NaN
df = df.dropna(subset=FEATURES)

# Drop rows where bus_V is invalid (below 2V = not stable)
df = df[df["bus_V"] > 2.0]

# Drop rows where lux is negative
df = df[df["lux"] >= 0]

df = df.reset_index(drop=True)
df.to_csv(CLEAN_CSV, index=False)
print(f"Cleaned log written to: {CLEAN_CSV}")
print(f"Rows in clean log: {len(df)}")

X = df[FEATURES].values.astype(np.float32)
print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)
print("StandardScaler fitted.")

# ── Build autoencoder (4→6→3→6→4 for 4 features) ─────────────────────────────
inp  = tf.keras.Input(shape=(NUM_FEATURES,))
enc1 = tf.keras.layers.Dense(6, activation="relu", name="enc1")(inp)
bot  = tf.keras.layers.Dense(3, activation="relu", name="bottleneck")(enc1)
dec1 = tf.keras.layers.Dense(6, activation="relu", name="dec1")(bot)
out  = tf.keras.layers.Dense(NUM_FEATURES, activation="linear", name="out")(dec1)

model = tf.keras.Model(inputs=inp, outputs=out, name="node2_local_ae")
model.compile(optimizer="adam", loss="mse")
print(model.summary())

# ── Train ──────────────────────────────────────────────────────────────────────
history = model.fit(
    X_scaled, X_scaled,
    epochs=300,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
print("Training done.")
print(f"Final train loss: {history.history['loss'][-1]}")
print(f"Final val   loss: {history.history['val_loss'][-1]}")

# ── Threshold ──────────────────────────────────────────────────────────────────
recon = model.predict(X_scaled, verbose=0)
errors = np.mean(np.square(X_scaled - recon), axis=1)

print("\nError stats on full log:")
print(f"  min:  {errors.min()}")
print(f"  max:  {errors.max()}")
print(f"  mean: {errors.mean()}")
print(f"  std:  {errors.std()}")

threshold_95 = float(np.percentile(errors, 95))
print(f"95th percentile threshold: {threshold_95}")

# ── Save results JSON ──────────────────────────────────────────────────────────
results = {
    "node": "node2",
    "features": FEATURES,
    "num_features": NUM_FEATURES,
    "threshold_95": threshold_95,
    "error_min":  float(errors.min()),
    "error_max":  float(errors.max()),
    "error_mean": float(errors.mean()),
    "error_std":  float(errors.std()),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std":  scaler.scale_.tolist(),
    "train_samples": int(X.shape[0])
}
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved results to {RESULTS_JSON}")

# ── Convert to TFLite INT8 ─────────────────────────────────────────────────────
def representative_dataset():
    for i in range(len(X_scaled)):
        yield [X_scaled[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
print(f"\nTFLite INT8 model size (bytes): {len(tflite_model)}")

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"Saved TFLite model to {TFLITE_PATH}")

# ── Write Arduino model header ─────────────────────────────────────────────────
with open(MODEL_H, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"// Node 2 local AE INT8 TFLite model — {len(tflite_model)} bytes\n")
    f.write(f"const unsigned int node2_local_ae_int8_model_len = {len(tflite_model)};\n")
    f.write("alignas(8) const unsigned char node2_local_ae_int8_model[] = {\n  ")
    hex_vals = [f"0x{b:02x}" for b in tflite_model]
    for i, h in enumerate(hex_vals):
        f.write(h)
        if i < len(hex_vals) - 1:
            f.write(", ")
        if (i + 1) % 12 == 0:
            f.write("\n  ")
    f.write("\n};\n")
print(f"Wrote model header to {MODEL_H}")

# ── Write Arduino scaler header ────────────────────────────────────────────────
mean_vals = scaler.mean_.tolist()
std_vals  = scaler.scale_.tolist()

with open(SCALER_H, "w") as f:
    f.write("#pragma once\n\n")
    f.write("// Node 2 StandardScaler parameters (4 features)\n")
    f.write(f"// Features: {', '.join(FEATURES)}\n\n")
    f.write(f"const int NODE2_NUM_FEATURES = {NUM_FEATURES};\n\n")
    f.write("const float NODE2_FEATURE_MEAN[] = {\n")
    for v in mean_vals:
        f.write(f"  {v:.8f}f,\n")
    f.write("};\n\n")
    f.write("const float NODE2_FEATURE_STD[] = {\n")
    for v in std_vals:
        f.write(f"  {v:.8f}f,\n")
    f.write("};\n")
print(f"Wrote scaler header to {SCALER_H}")

print("\n=== Done. Copy this threshold into Arduino code: ===")
print(f"NODE2_LOCAL_AE_THRESHOLD = {threshold_95}")
