// AEGIS Node 1 - Local 3-feature Autoencoder
// Features: temp_C, bus_V, current_mA_abs
// Model: node1_local_ae_int8 (trained on node1_log.csv)

#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_INA219.h>

#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

#include "node1_local_ae_int8.h"
#include "aegis_node1_scaler.h"

#define ONE_WIRE_BUS 4
#define LED_PIN 2

// 1 = LOG RAW CSV ONLY (for training)
// 0 = RUN AE + PRINT MSE + DECISION (for demo)
#define AEGIS_LOG_MODE 0

Adafruit_INA219 ina219;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds18b20(&oneWire);

const int kNumFeatures = 3;
constexpr int kTensorArenaSize = 16 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

float feature_mean[kNumFeatures];
float feature_std[kNumFeatures];

// This value will be updated after you retrain
const float NODE1_LOCAL_AE_THRESHOLD = 3.61583066f;

// ------------ Helper functions ------------

void init_scaler() {
  for (int i = 0; i < kNumFeatures; i++) {
    feature_mean[i] = NODE1_FEATURE_MEAN[i];
    feature_std[i]  = NODE1_FEATURE_STD[i];
    if (feature_std[i] == 0.0f) {
      feature_std[i] = 1.0f;
    }
  }
}

void copy_features_to_input_tensor(float* features_scaled) {
  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumFeatures; i++) {
      input->data.f[i] = features_scaled[i];
    }
  } else if (input->type == kTfLiteInt8) {
    float scale = input->params.scale;
    int zero_point = input->params.zero_point;
    for (int i = 0; i < kNumFeatures; i++) {
      float x = features_scaled[i];
      int32_t q = static_cast<int32_t>(roundf(x / scale)) + zero_point;
      if (q < -128) q = -128;
      if (q > 127)  q = 127;
      input->data.int8[i] = static_cast<int8_t>(q);
    }
  } else {
    Serial.println("Unsupported input tensor type.");
  }
}

void copy_output_to_float(float* recon_scaled) {
  if (output->type == kTfLiteFloat32) {
    for (int i = 0; i < kNumFeatures; i++) {
      recon_scaled[i] = output->data.f[i];
    }
  } else if (output->type == kTfLiteInt8) {
    float scale = output->params.scale;
    int zero_point = output->params.zero_point;
    for (int i = 0; i < kNumFeatures; i++) {
      int8_t q = output->data.int8[i];
      recon_scaled[i] = (static_cast<int32_t>(q) - zero_point) * scale;
    }
  } else {
    Serial.println("Unsupported output tensor type.");
  }
}

float compute_mse(const float* a, const float* b) {
  float sum = 0.0f;
  for (int i = 0; i < kNumFeatures; i++) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum / (float)kNumFeatures;
}

// ------------ Setup ------------

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);

  Wire.begin(21, 22);

  if (!ina219.begin()) {
    Serial.println("INA219 not found.");
    while (true) { delay(10); }
  }

  ds18b20.begin();

  init_scaler();

  model = tflite::GetModel(node1_local_ae_int8_model);

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed.");
    while (true) { delay(10); }
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

#if AEGIS_LOG_MODE
  Serial.println("timestamp_ms,temp_C,bus_V,current_mA_abs,power_mW");
#else
  Serial.println("Node 1 local AE ready.");
  Serial.print("Input type: ");
  Serial.println(input->type == kTfLiteFloat32 ? "FLOAT32" :
                 input->type == kTfLiteInt8    ? "INT8" : "OTHER");
  Serial.print("Threshold (scaled MSE): ");
  Serial.println(NODE1_LOCAL_AE_THRESHOLD, 6);
#endif
}

// ------------ Loop ------------

void loop() {
  ds18b20.requestTemperatures();
  float tempC = ds18b20.getTempCByIndex(0);

  // DS18B20 returns -127 when disconnected/not responding, so skip those rows. [web:895]
  if (tempC <= -100.0f) {
    delay(3000);
    return;
  }

  float busV = ina219.getBusVoltage_V();
  float current_raw = ina219.getCurrent_mA();
  float current_abs = fabs(current_raw);
  float power_mW = ina219.getPower_mW();

  float features_raw[kNumFeatures];
  features_raw[0] = tempC;
  features_raw[1] = busV;
  features_raw[2] = current_abs;

  float features_scaled[kNumFeatures];
  for (int i = 0; i < kNumFeatures; i++) {
    features_scaled[i] = (features_raw[i] - feature_mean[i]) / feature_std[i];
  }

#if AEGIS_LOG_MODE
  // Print ONLY the CSV line needed by your training cleaner.
  Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f\n",
                (unsigned long)millis(),
                tempC,
                busV,
                current_abs,
                power_mW);
#else
  copy_features_to_input_tensor(features_scaled);

  TfLiteStatus s = interpreter->Invoke();
  if (s != kTfLiteOk) {
    Serial.println("Invoke failed.");
    delay(3000);
    return;
  }

  float recon_scaled[kNumFeatures];
  copy_output_to_float(recon_scaled);

  float mse = compute_mse(features_scaled, recon_scaled);
  bool isAnomaly = (mse > NODE1_LOCAL_AE_THRESHOLD);

  Serial.println("----------------------------------------");
  Serial.print("temp_C: "); Serial.println(tempC, 4);
  Serial.print("bus_V: "); Serial.println(busV, 4);
  Serial.print("current_mA_abs: "); Serial.println(current_abs, 4);
  Serial.print("power_mW: "); Serial.println(power_mW, 4);
  Serial.print("MSE_scaled: "); Serial.println(mse, 6);
  Serial.print("Decision: "); Serial.println(isAnomaly ? "ANOMALY" : "NORMAL");

  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);
#endif

  delay(3000);
}
