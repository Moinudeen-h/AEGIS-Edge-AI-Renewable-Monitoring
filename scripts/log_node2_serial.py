import os
import serial

base_dir = r"C:\Users\MOINODHEEN\Moinu\AEGIS_Project"
out_path = os.path.join(base_dir, "data", "logs", "node2_log.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

SERIAL_PORT = "COM4"   # change if Node 2 is on a different COM port
BAUD_RATE   = 115200

print(f"Opening {SERIAL_PORT} at {BAUD_RATE}...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

print(f"Logging to: {out_path}")
print("Press Ctrl+C to stop.\n")

with open(out_path, "w", encoding="utf-8") as f:
    f.write("timestamp_ms,bus_V,current_mA_abs,power_mW,lux\n")  # write header first
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            # Accept only valid CSV rows: starts with digit, has exactly 4 commas
            if line[0].isdigit() and line.count(",") == 4:
                print(line)
                f.write(line + "\n")
                f.flush()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        print("Serial closed.")
