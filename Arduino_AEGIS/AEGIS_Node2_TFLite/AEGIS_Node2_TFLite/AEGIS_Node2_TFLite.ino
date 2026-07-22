#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <BH1750.h>

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "node2_local_ae_int8.h"
#include "aegis_node2_scaler.h"

#define I2C_SDA 25
#define I2C_SCL 26
#define INA219_ADDR 0x40
#define BH1750_ADDR 0x23
#define LED_PIN 33

// 1 = LOG CSV ONLY (for training data collection)
// 0 = RUN AE INFERENCE (anomaly detection demo)
#define AEGIS_LOG_MODE 0

// ── Update this after training ─────────────────────────────────────────────────
const float NODE2_LOCAL_AE_THRESHOLD = 1.1032283f;

const int kNumFeatures = 4;
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

Adafruit_INA219 ina219(INA219_ADDR);
BH1750 lightMeter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

float feature_mean[kNumFeatures];
float feature_std[kNumFeatures];

// ── Helpers ────────────────────────────────────────────────────────────────────
void init_scaler() {
  for (int i = 0; i < kNumFeatures; i++) {
    feature_mean[i] = NODE2_FEATURE_MEAN[i];
    feature_std[i]  = NODE2_FEATURE_STD[i];
    if (feature_std[i] == 0.0f) feature_std[i] = 1.0f;
  }
}

void copy_to_input(float* scaled) {
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumFeatures; i++) input->data.f[i] = scaled[i];
  } else if (input->type == kTfLiteInt8) {
    float sc = input->params.scale;
    int   zp = input->params.zero_point;
    for (int i = 0; i < kNumFeatures; i++) {
      int32_t q = (int32_t)roundf(scaled[i] / sc) + zp;
      q = q < -128 ? -128 : (q > 127 ? 127 : q);
      input->data.int8[i] = (int8_t)q;
    }
  }
}

void copy_from_output(float* recon) {
  if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumFeatures; i++) recon[i] = output->data.f[i];
  } else if (output->type == kTfLiteInt8) {
    float sc = output->params.scale;
    int   zp = output->params.zero_point;
    for (int i = 0; i < kNumFeatures; i++)
      recon[i] = ((int32_t)output->data.int8[i] - zp) * sc;
  }
}

float compute_mse(float* a, float* b) {
  float s = 0.0f;
  for (int i = 0; i < kNumFeatures; i++) { float d = a[i]-b[i]; s += d*d; }
  return s / kNumFeatures;
}

// ── Setup ──────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(800);
  pinMode(LED_PIN, OUTPUT);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not found."); while (true) { delay(50); }
  }
  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, BH1750_ADDR, &Wire)) {
    Serial.println("ERROR: BH1750 not found."); while (true) { delay(50); }
  }

  init_scaler();

  model = tflite::GetModel(node2_local_ae_int8_model);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed."); while (true) { delay(50); }
  }
  input  = interpreter->input(0);
  output = interpreter->output(0);

#if AEGIS_LOG_MODE
  Serial.println("timestamp_ms,bus_V,current_mA_abs,power_mW,lux");
#else
  Serial.println("Node 2 local AE ready.");
  Serial.print("Threshold: "); Serial.println(NODE2_LOCAL_AE_THRESHOLD, 6);
#endif
}

// ── Loop ───────────────────────────────────────────────────────────────────────
void loop() {
  float busV        = ina219.getBusVoltage_V();
  float current_abs = fabs(ina219.getCurrent_mA());
  float power_mW    = ina219.getPower_mW();
  float lux         = lightMeter.readLightLevel();
  if (lux < 0) lux  = 0.0f;

  if (busV < 2.0f) { delay(3000); return; }

  float raw[kNumFeatures]    = { busV, current_abs, power_mW, lux };
  float scaled[kNumFeatures];
  for (int i = 0; i < kNumFeatures; i++)
    scaled[i] = (raw[i] - feature_mean[i]) / feature_std[i];

#if AEGIS_LOG_MODE
  Serial.printf("%lu,%.4f,%.4f,%.4f,%.2f\n",
                (unsigned long)millis(),
                busV, current_abs, power_mW, lux);
#else
  copy_to_input(scaled);
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed."); delay(3000); return;
  }
  float recon[kNumFeatures];
  copy_from_output(recon);

  float mse       = compute_mse(scaled, recon);
  bool isAnomaly  = (mse > NODE2_LOCAL_AE_THRESHOLD);

  Serial.println("----------------------------------------");
  Serial.print("bus_V: ");        Serial.println(busV, 4);
  Serial.print("current_mA_abs: "); Serial.println(current_abs, 4);
  Serial.print("power_mW: ");     Serial.println(power_mW, 4);
  Serial.print("lux: ");          Serial.println(lux, 2);
  Serial.print("MSE_scaled: ");   Serial.println(mse, 6);
  Serial.print("Decision: ");     Serial.println(isAnomaly ? "ANOMALY" : "NORMAL");

  digitalWrite(LED_PIN, HIGH); delay(200); digitalWrite(LED_PIN, LOW);
#endif

  delay(3000);
}
