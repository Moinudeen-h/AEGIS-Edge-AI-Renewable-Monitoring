import os
import serial

base_dir = r"C:\Users\MOINODHEEN\Moinu\AEGIS_Project"
out_path = os.path.join(base_dir, "data", "logs", "node1_log.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

SERIAL_PORT = "COM3"
BAUD_RATE   = 115200

print(f"Opening {SERIAL_PORT} at {BAUD_RATE}...")
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

print(f"APPENDING to: {out_path}")
print("Press Ctrl+C to stop.\n")

# Open in APPEND mode ("a") — this is the key difference from log_node1_serial.py
with open(out_path, "a", encoding="utf-8") as f:
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line[0].isdigit() and line.count(",") == 4:
                print(line)
                f.write(line + "\n")
                f.flush()
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        print("Serial closed.")
