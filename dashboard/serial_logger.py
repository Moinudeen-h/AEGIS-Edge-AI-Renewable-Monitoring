# ============================================================
# AEGIS Serial Logger — serial_logger.py (CSV FIXED VERSION)
# AEGIS: Autonomous Energy Grid Intelligence System
# UWE Bristol Final Year Project — 24040034
#
# FIXED: Parses CSV directly from ESP32 (your format)
# Run Node 1:  python serial_logger.py --node 1 --port COM3 --debug
# Run Node 2:  python serial_logger.py --node 2 --port COM4 --debug
# ============================================================

import serial
import serial.tools.list_ports
import csv
import os
import sys
import time
import argparse
from datetime import datetime

# ─────────────────────────────────────────────
# DEFAULTS  (overridden by command-line args)
# ─────────────────────────────────────────────
DEFAULT_NODE      = 2
DEFAULT_COM_NODE1 = "COM3"
DEFAULT_COM_NODE2 = "COM4"
BAUD_RATE         = 115200
RETRY_DELAY       = 5
BLOCK_TIMEOUT     = 10.0  # Increased for stability

# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="AEGIS Serial Logger — reads ESP32 CSV and writes CSV"
    )
    parser.add_argument(
        "--node", type=int, choices=[1, 2], default=DEFAULT_NODE,
        help="Which node to log (1 or 2). Default: 1"
    )
    parser.add_argument(
        "--port", type=str, default=None,
        help="COM port, e.g. COM3 or /dev/ttyUSB0. "
             "Defaults to COM3 for Node 1, COM4 for Node 2."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print every raw line (diagnosis)"
    )
    return parser.parse_args()

# ─────────────────────────────────────────────
# CSV HELPERS
# ─────────────────────────────────────────────
def ensure_csv(path, headers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(headers)
        print(f"[LOGGER] Created CSV: {path}")

def append_row(path, row, headers):
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=headers, extrasaction="ignore").writerow(row)

# ─────────────────────────────────────────────
# CSV PARSER (NEW — works with your ESP32 format)
# ─────────────────────────────────────────────
def parse_csv_line(line, node, debug=False):
    """Parse CSV line from ESP32: timestamp_ms,temp_C,bus_V,current_mA_abs,power_mW,mse_scaled,decision"""
    if ',' not in line or line.startswith("timestamp_ms,"):  # Skip header
        if debug:
            print(f"[CSV] Header/Skipped: {line.strip()}")
        return None
    
    try:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 7:
            if debug:
                print(f"[CSV] Too few columns: {len(parts)} — {line.strip()}")
            return None
        
        timestamp_ms = int(parts[0])
        
        if node == 1:  # Node 1: timestamp_ms,temp_C,bus_V,current_mA_abs,power_mW,mse_scaled,decision
            temp_C = float(parts[1])
            bus_V = float(parts[2])
            current_mA_abs = abs(float(parts[3]))
            power_mW = float(parts[4])
            mse_scaled = float(parts[5])
            decision = parts[6].strip().upper()
            
            row = {
                "timestamp_ms": int(time.time() * 1000),
                "temp_C": temp_C,
                "bus_V": bus_V,
                "current_mA_abs": current_mA_abs,
                "power_mW": power_mW,
                "mse_scaled": mse_scaled,
                "decision": decision
            }
            
        else:  # Node 2: timestamp_ms,bus_V,current_mA_abs,power_mW,lux,mse_scaled,decision
            bus_V = float(parts[1])
            current_mA_abs = abs(float(parts[2]))
            power_mW = float(parts[3])
            lux = float(parts[4])
            mse_scaled = float(parts[5])
            decision = parts[6].strip().upper()
            
            row = {
                "timestamp_ms": int(time.time() * 1000),
                "lux": lux,
                "bus_V": bus_V,
                "current_mA_abs": current_mA_abs,
                "power_mW": power_mW,
                "mse_scaled": mse_scaled,
                "decision": decision
            }
        
        return row
        
    except (ValueError, IndexError) as e:
        if debug:
            print(f"[CSV] Parse error: {e} — {line.strip()}")
        return None

# ─────────────────────────────────────────────
# SERIAL
# ─────────────────────────────────────────────
def open_serial(port):
    try:
        ser = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=2.0)
        print(f"[LOGGER] Connected to {port} @ {BAUD_RATE} baud ✓")
        time.sleep(2)  # Stabilize
        return ser
    except serial.SerialException as e:
        print(f"[LOGGER] Could not open {port}: {e}")
        return None

def list_ports():
    ports = serial.tools.list_ports.comports()
    if ports:
        print("[LOGGER] Available ports:")
        for p in ports:
            print(f"         {p.device} — {p.description}")
    else:
        print("[LOGGER] No serial ports.")

# ─────────────────────────────────────────────
# MAIN LOOP (CSV VERSION)
# ─────────────────────────────────────────────
def run_logger(node, port, csv_path, csv_headers, debug):
    ensure_csv(csv_path, csv_headers)
    print(f"[LOGGER] Node {node} CSV | Output: {csv_path}")
    list_ports()
    if debug: print("[DEBUG] ON — shows all raw lines\n")

    row_count = 0

    while True:
        ser = open_serial(port)
        if ser is None:
            print(f"[LOGGER] Retry {port} in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
            continue

        try:
            while True:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                if debug:
                    print(f"[RAW] '{line}'")

                # PARSE CSV DIRECTLY
                row = parse_csv_line(line, node, debug)
                if row:
                    append_row(csv_path, row, csv_headers)
                    row_count += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    
                    sensor = row.get('temp_C', '?') if node == 1 else row.get('lux', '?')
                    print(f"[{ts}] N{node} #{row_count:>3} | "
                          f"{'temp={:.2f}°C'.format(sensor) if node==1 else 'lux={:.1f}'.format(sensor)} | "
                          f"V={row['bus_V']:.3f} | "
                          f"I={row['current_mA_abs']:.1f}mA | "
                          f"MSE={row['mse_scaled']:.6f} | {row['decision']}")

        except KeyboardInterrupt:
            print(f"\n[LOGGER] Node {node} stopped. Rows saved: {row_count}")
            sys.exit(0)
        finally:
            try:
                ser.close()
            except:
                pass
            print(f"[LOGGER] Disconnected. Retry in {RETRY_DELAY}s...")

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    node = args.node
    port = args.port or (DEFAULT_COM_NODE1 if node == 1 else DEFAULT_COM_NODE2)

    if node == 1:
        csv_path = os.path.join("data", "logs", "node1_log.csv")
        csv_headers = ["timestamp_ms", "temp_C", "bus_V", "current_mA_abs", "power_mW", "mse_scaled", "decision"]
    else:
        csv_path = os.path.join("data", "logs", "node2_log.csv")
        csv_headers = ["timestamp_ms", "lux", "bus_V", "current_mA_abs", "power_mW", "mse_scaled", "decision"]

    print("=" * 70)
    print(f"  AEGIS CSV LOGGER ✓ | Node {node} | {port} → {csv_path}")
    print("=" * 70)

    run_logger(node, port, csv_path, csv_headers, args.debug)