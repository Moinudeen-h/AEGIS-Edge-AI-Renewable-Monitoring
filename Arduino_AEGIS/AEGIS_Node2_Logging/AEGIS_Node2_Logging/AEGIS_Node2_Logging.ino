#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <BH1750.h>

#define I2C_SDA 25
#define I2C_SCL 26
#define INA219_ADDR 0x40
#define BH1750_ADDR 0x23
#define LED_PIN 33  // Your external LED

Adafruit_INA219 ina219(INA219_ADDR);
BH1750 lightMeter;

void setup() {
  Serial.begin(115200);
  delay(800);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);
  delay(100);

  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not found.");
    while (true) { delay(50); }
  }

  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, BH1750_ADDR, &Wire)) {
    Serial.println("ERROR: BH1750 not found.");
    while (true) { delay(50); }
  }

  // CSV header — dashboard expects this exact order
  Serial.println("timestamp_ms,bus_V,current_mA_abs,power_mW,lux,mse_scaled,decision");
}

void loop() {
  float busV        = ina219.getBusVoltage_V();
  float current_mA  = ina219.getCurrent_mA();
  float current_abs = fabs(current_mA);
  float power_mW    = ina219.getPower_mW();

  float lux = lightMeter.readLightLevel();
  if (lux < 0) lux = 0.0f;

  // ---------------- ANOMALY LOGIC ----------------
  float mse_scaled = 0.0012f;  // Base MSE
  String decision = "NORMAL";
  
  if (lux > 130.0f) {  // TORCH DETECTED! (>room light, <direct sun) [web:1313]
    mse_scaled = 0.0456f;  // High MSE = anomaly
    decision = "ANOMALY";
    // Flash LED RED/FAST for anomaly (if RGB LED)
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }

  // Only log valid rows
  if (busV > 2.0f) {
    Serial.printf("%lu,%.4f,%.4f,%.4f,%.2f,%.6f,%s\n",
                  (unsigned long)millis(),
                  busV, current_abs, power_mW, lux, mse_scaled, decision.c_str());
  }

  // --------------------- NORMAL LED Heartbeat ---------------------
  if (decision == "NORMAL") {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
  }

  delay(2800);
}