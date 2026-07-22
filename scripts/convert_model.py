import os

# Input model
tflite_path = r"C:\Users\MOINODHEEN\Moinu\AEGIS_Project\models\tflite\wind_farm_a_int8.tflite"

# Output header goes directly into the Arduino sketch folder
output_path = r"C:\Users\MOINODHEEN\Moinu\AEGIS_Project\Arduino_AEGIS\AEGIS_Node1_TFLite\AEGIS_Node1_TFLite\wind_farm_a_int8.h"

with open(tflite_path, "rb") as f:
    model_bytes = f.read()

model_size = len(model_bytes)
print(f"Model size: {model_size} bytes")

hex_lines = []
for i in range(0, model_size, 12):
    chunk = model_bytes[i:i+12]
    hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
    hex_lines.append("  " + hex_str)

hex_body = ",\n".join(hex_lines)

header_content = f"""// Auto-generated from wind_farm_a_int8.tflite
// Wind Farm A Autoencoder INT8 - AEGIS Node 1
// Input: 81 features, Output: 81 reconstructed features

#ifndef WIND_FARM_A_INT8_H
#define WIND_FARM_A_INT8_H

const unsigned char wind_farm_a_int8_model[] = {{
{hex_body}
}};

const unsigned int wind_farm_a_int8_model_len = {model_size};

#endif
"""

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    f.write(header_content)

print(f"Header written to:\n  {output_path}")
print("Done.")
