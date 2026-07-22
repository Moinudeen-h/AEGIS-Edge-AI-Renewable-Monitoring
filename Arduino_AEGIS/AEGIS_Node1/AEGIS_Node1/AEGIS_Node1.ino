#include <Wire.h>
#include <Adafruit_INA219.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// --------------------- Pin Definitions ---------------------
#define ONE_WIRE_BUS 4       // GPIO pin connected to DS18B20 data line
#define LED_PIN 2            // Built-in LED on most ESP32 DevKit boards

// --------------------- Sensor Objects ----------------------
Adafruit_INA219 ina219;      // Object to interface with INA219 current/voltage sensor
OneWire oneWire(ONE_WIRE_BUS); 
DallasTemperature ds18b20(&oneWire); // Object for DS18B20 temperature sensor

void setup() {
  // Initialize serial communication for debugging and logging
  Serial.begin(115200);
  delay(1000); // Allow time for ESP32 to stabilize

  // --------------------- LED Setup --------------------------
  pinMode(LED_PIN, OUTPUT); // Configure built-in LED as output

  // --------------------- I2C Setup --------------------------
  Wire.begin(21, 22);       // SDA = GPIO21, SCL = GPIO22

  // --------------------- INA219 Setup -----------------------
  if (!ina219.begin()) {
    Serial.println("INA219 not found. Check SDA, SCL, VCC and GND.");
    while (1) { delay(10); } // Stop here if INA219 not detected
  }

  // --------------------- DS18B20 Setup ---------------------
  ds18b20.begin();          // Initialize DS18B20 sensor

  // Verify at least one DS18B20 is detected
  if (ds18b20.getDeviceCount() == 0) {
    Serial.println("DS18B20 not found. Check GPIO4, GND, VDD, and 4.7k pull-up resistor.");
  }

  // --------------------- Startup Message -------------------
  Serial.println("AEGIS Node 1 - Initialization complete");
  Serial.println("Sensors: INA219 (power monitor) + DS18B20 (temperature)");
}

void loop() {
  // --------------------- Temperature Measurement -----------------
  ds18b20.requestTemperatures();                // Trigger temperature conversion
  float tempC = ds18b20.getTempCByIndex(0);   // Read temperature of first sensor

  // --------------------- Power Measurement ----------------------
  float busVoltage = ina219.getBusVoltage_V(); // Voltage on the bus
  float current_mA = ina219.getCurrent_mA();  // Current in milliamps
  float power_mW   = ina219.getPower_mW();    // Power in milliwatts

  // --------------------- Print Results -------------------------
  Serial.println("----------------------------------------");

  // Temperature check: -127°C means sensor not detected
  if (tempC == -127.0) {
    Serial.println("Temperature (C): ERROR - sensor not detected");
  } else {
    Serial.print("Temperature (C): ");
    Serial.println(tempC, 2); // Print with 2 decimal places
  }

  Serial.print("Bus voltage (V):  "); Serial.println(busVoltage, 3);
  Serial.print("Current (mA):     "); Serial.println(current_mA, 3);
  Serial.print("Power (mW):       "); Serial.println(power_mW, 3);

  // --------------------- LED Indicator ------------------------
  // Blink built-in LED to indicate one loop iteration completed
  digitalWrite(LED_PIN, HIGH); // Turn LED on
  delay(200);                  // Keep LED on for 200 ms
  digitalWrite(LED_PIN, LOW);  // Turn LED off

  delay(2800); // Remaining delay to make total loop ~3 seconds
}