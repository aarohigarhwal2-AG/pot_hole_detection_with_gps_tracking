"""
db_manager.py - Persistent database and file logger for detected pothole incidents.
"""
import os
import sqlite3
import csv
from datetime import datetime
import cv2


class DatabaseManager:
    def __init__(self, db_path: str = "potholes.db", csv_path: str = "pothole_log.csv", img_dir: str = "detections"):
        self.db_path = db_path
        self.csv_path = csv_path
        self.img_dir = img_dir

        os.makedirs(self.img_dir, exist_ok=True)
        self._init_db()
        self._init_csv()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS potholes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pothole_uid INTEGER,
                    timestamp TEXT,
                    latitude REAL,
                    longitude REAL,
                    confidence REAL,
                    severity TEXT,
                    image_path TEXT
                )
            """)
            conn.commit()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Pothole_UID", "Timestamp", "Latitude", "Longitude", "Confidence", "Severity", "Image_Path"])

    def log_pothole(self, pothole_uid: int, lat: float, lon: float, confidence: float,
                    severity: str, frame, bbox: tuple) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_filename = f"pothole_{pothole_uid}_{time_tag}.jpg"
        img_full_path = os.path.join(self.img_dir, img_filename)

        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        x1_c, y1_c = max(0, x1 - 20), max(0, y1 - 20)
        x2_c, y2_c = min(w, x2 + 20), min(h, y2 + 20)
        crop = frame[y1_c:y2_c, x1_c:x2_c]
        if crop.size > 0:
            cv2.imwrite(img_full_path, crop)
        else:
            cv2.imwrite(img_full_path, frame)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO potholes (pothole_uid, timestamp, latitude, longitude, confidence, severity, image_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pothole_uid, timestamp, lat, lon, confidence, severity, img_full_path))
            conn.commit()

        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([pothole_uid, pothole_uid, timestamp, lat, lon, f"{confidence:.2f}", severity, img_full_path])

        return img_full_path

    def get_all_potholes(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM potholes ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]
