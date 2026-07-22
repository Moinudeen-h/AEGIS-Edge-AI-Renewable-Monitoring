#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_INA219.h>
#include <BH1750.h>

#define I2C_SDA 25
#define I2C_SCL 26

#define INA219_ADDR 0x40
#define BH1750_ADDR 0x23
#define LED_PIN 33

Adafruit_INA219 ina219(INA219_ADDR);
BH1750 lightMeter;

void i2c_scan() {
  Serial.println("I2C scan starting...");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      Serial.print("Found I2C device at 0x");
      if (addr < 16) Serial.print("0");
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) Serial.println("No I2C devices found. Check wiring/power.");
  Serial.println("I2C scan done.\n");
}

void setup() {
  Serial.begin(115200);
  delay(800);

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.println("\n=== AEGIS Node 2 Sensor Debug ===");

  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);
  delay(100);

  i2c_scan();

  Serial.println("Initializing INA219...");
  if (!ina219.begin()) {
    Serial.println("ERROR: INA219 not found. Check wiring/address.");
    while (true) { delay(50); }
  }
  Serial.println("INA219 OK.");

  Serial.println("Initializing BH1750...");
  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE, BH1750_ADDR, &Wire)) {
    Serial.println("ERROR: BH1750 not found. Check wiring/address/ADDR pin.");
    while (true) { delay(50); }
  }
  Serial.println("BH1750 OK.\n");

  Serial.println("timestamp_ms,bus_V,shunt_mV,vinplus_V,current_mA_abs,power_mW,lux");
}

void loop() {
  float shunt_mV = ina219.getShuntVoltage_mV();
  float busV = ina219.getBusVoltage_V();
  float vinplusV = busV + (shunt_mV / 1000.0f);

  float current_mA = ina219.getCurrent_mA();
  float current_abs = fabs(current_mA);
  float power_mW = ina219.getPower_mW();

  float lux = lightMeter.readLightLevel();
  if (lux < 0) lux = -1.0f;

  Serial.printf("%lu,%.4f,%.3f,%.4f,%.4f,%.4f,%.2f\n",
                (unsigned long)millis(),
                busV,
                shunt_mV,
                vinplusV,
                current_abs,
                power_mW,
                lux);

  digitalWrite(LED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_PIN, LOW);

  delay(2800);
}
