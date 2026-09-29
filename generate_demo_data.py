"""
generate_demo_data.py - Creates a synthetic road video and initial database seeds for testing.
"""
import cv2
import numpy as np
import random
from db_manager import DatabaseManager


def generate_synthetic_video(filename: str = "demo_road.mp4", duration_sec: int = 15, fps: int = 30):
    print(f"[Generator] Generating synthetic road video '{filename}'...")
    w, h = 640, 360
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    total_frames = duration_sec * fps
    pothole_frames = [int(total_frames * 0.2), int(total_frames * 0.5), int(total_frames * 0.75)]

    for f in range(total_frames):
        # Create asphalt road background
        frame = np.full((h, w, 3), 60, dtype=np.uint8)

        # Lane markings
        dash_offset = (f * 10) % 60
        for y in range(0, h, 60):
            cv2.rectangle(frame, (w // 2 - 4, y + dash_offset), (w // 2 + 4, y + dash_offset + 30), (255, 255, 255), -1)

        # Road edges
        cv2.line(frame, (60, 0), (60, h), (200, 200, 200), 4)
        cv2.line(frame, (w - 60, 0), (w - 60, h), (200, 200, 200), 4)

        # Draw moving simulated potholes
        for pf in pothole_frames:
            if pf <= f < pf + 40:
                progress = (f - pf) / 40.0
                py = int(h * progress)
                px = int(w * 0.35 if pf % 2 == 0 else w * 0.65)
                radius_x = int(25 * (progress + 0.5))
                radius_y = int(12 * (progress + 0.5))
                cv2.ellipse(frame, (px, py), (radius_x, radius_y), 0, 0, 360, (20, 20, 20), -1)
                cv2.ellipse(frame, (px, py), (radius_x + 2, radius_y + 2), 0, 0, 360, (40, 40, 40), 2)

        out.write(frame)

    out.release()
    print(f"[Generator] Finished creating '{filename}'.")


def seed_sample_database():
    print("[Generator] Seeding sample pothole coordinates into database...")
    db = DatabaseManager()
    base_lat, base_lon = 28.6139, 77.2090
    dummy_frame = np.full((200, 200, 3), 40, dtype=np.uint8)

    severities = ["Minor", "Moderate", "Severe"]
    for i in range(1, 6):
        d_lat = base_lat + (random.uniform(-0.015, 0.015))
        d_lon = base_lon + (random.uniform(-0.015, 0.015))
        sev = random.choice(severities)
        conf = random.uniform(0.65, 0.95)
        db.log_pothole(i, d_lat, d_lon, conf, sev, dummy_frame, (20, 20, 180, 180))

    print("[Generator] Sample incidents added to database.")


if __name__ == "__main__":
    generate_synthetic_video()
    seed_sample_database()
