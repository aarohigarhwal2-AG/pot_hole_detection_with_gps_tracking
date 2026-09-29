"""
gps_tracker.py - GPS reader supporting both real NMEA serial hardware and simulated routes.
"""
import time
import math
from typing import Tuple


class GPSTracker:
    def __init__(self, mode: str = "simulated", port: str = "COM3", baudrate: int = 9600,
                 start_lat: float = 28.6139, start_lon: float = 77.2090):
        self.mode = mode
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None

        self.sim_lat = start_lat
        self.sim_lon = start_lon
        self.sim_heading = 45.0
        self.sim_speed_kmh = 35.0
        self.last_sim_time = time.time()

        if self.mode == "serial":
            try:
                import serial
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
                print(f"[GPS] Connected to hardware GPS on {self.port}.")
            except Exception as e:
                print(f"[GPS Warning] Serial connection failed ({e}). Defaulting to simulated GPS.")
                self.mode = "simulated"

    def get_coordinates(self) -> Tuple[float, float, float]:
        if self.mode == "serial" and self.serial_conn and self.serial_conn.is_open:
            try:
                import pynmea2
                line = self.serial_conn.readline().decode('ascii', errors='replace').strip()
                if line.startswith(('$GPRMC', '$GNRMC')):
                    msg = pynmea2.parse(line)
                    if msg.status == 'A':
                        speed_kmh = float(msg.spd_over_grnd or 0.0) * 1.852
                        return float(msg.latitude), float(msg.longitude), speed_kmh
            except Exception:
                pass

        now = time.time()
        dt = now - self.last_sim_time
        self.last_sim_time = now
        speed_mps = (self.sim_speed_kmh * 1000) / 3600
        dist = speed_mps * dt
        r_earth = 6378137.0
        d_lat = (dist * math.cos(math.radians(self.sim_heading))) / r_earth
        d_lon = (dist * math.sin(math.radians(self.sim_heading))) / (r_earth * math.cos(math.radians(self.sim_lat)))
        self.sim_lat += math.degrees(d_lat)
        self.sim_lon += math.degrees(d_lon)
        self.sim_heading += 0.5 * math.sin(now / 5.0)
        return round(self.sim_lat, 6), round(self.sim_lon, 6), round(self.sim_speed_kmh, 1)

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
