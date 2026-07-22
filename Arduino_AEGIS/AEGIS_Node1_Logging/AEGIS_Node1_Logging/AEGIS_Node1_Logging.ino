#include <Arduino.h>
#include <Wire.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Adafruit_INA219.h>

#define ONE_WIRE_BUS 4
#define LED_PIN 2  // Built-in LED (change if different)

Adafruit_INA219 ina219;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds18b20(&oneWire);

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Wire.begin(21, 22);

  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not found");
    while (true) { delay(10); }
  }

  ds18b20.begin();

  // CSV header — dashboard expects this exact order
  Serial.println("timestamp_ms,temp_C,bus_V,current_mA_abs,power_mW,mse_scaled,decision");
}

void loop() {
  unsigned long t = millis();

  ds18b20.requestTemperatures();
  float tempC = ds18b20.getTempCByIndex(0);
  if (tempC == DEVICE_DISCONNECTED_C) tempC = -999.0f;

  float busV = ina219.getBusVoltage_V();
  float current_raw = ina219.getCurrent_mA();
  float current_abs = fabs(current_raw);
  float power_mW = ina219.getPower_mW();

  // Fake MSE/decision for demo (add real TFLite later)
  float mse_scaled = 0.0008f;  // Normal range
  String decision = "NORMAL";

  // Only log valid rows
  if (busV > 2.0f && tempC > -50.0f) {
    Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.6f,%s\n",
                  t, tempC, busV, current_abs, power_mW, mse_scaled, decision.c_str());
  }

  // --------------------- LED Heartbeat (demo visible) ---------------------
  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);

  delay(2800);  // Total 3 seconds, matches Node 2
}
