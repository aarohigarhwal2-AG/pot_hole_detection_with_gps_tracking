"""
detect_video.py - High-Accuracy Pothole Detector with ROI masking, CLAHE preprocessing, and GPS synchronization.
"""
import cv2
import numpy as np
import argparse
from gps_tracker import GPSTracker
from tracker import PotholeTracker
from db_manager import DatabaseManager


def apply_clahe_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def get_road_roi_mask(frame_shape):
    h, w, _ = frame_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([
        [int(w * 0.05), h],
        [int(w * 0.95), h],
        [int(w * 0.70), int(h * 0.45)],
        [int(w * 0.30), int(h * 0.45)]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts


def calculate_severity(bbox: tuple, frame_area: int, confidence: float) -> str:
    x1, y1, x2, y2 = bbox
    area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
    if area_ratio > 0.035 or (area_ratio > 0.02 and confidence > 0.85):
        return "Severe"
    elif area_ratio > 0.012 or confidence > 0.70:
        return "Moderate"
    return "Minor"


def run_detection(video_source: str = "demo_road.mp4", gps_mode: str = "simulated",
                  gps_port: str = "COM3", model_path: str = "weights/best.pt", conf_thresh: float = 0.48):
    print(f"[Init] Loading YOLO model: {model_path}...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        print(f"[Warning] Could not load YOLO model ({e}). Using mock detector.")
        model = None

    cap = cv2.VideoCapture(0 if video_source == "0" else video_source)
    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {video_source}")
        return

    gps = GPSTracker(mode=gps_mode, port=gps_port)
    tracker = PotholeTracker(min_confirmed_frames=3)
    db = DatabaseManager()

    print("[System] Pipeline running with ROI Mask + CLAHE + Multi-Frame Confirmation.")

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w, _ = frame.shape
        frame_area = h * w
        lat, lon, speed = gps.get_coordinates()

        enhanced_frame = apply_clahe_contrast(frame)
        roi_mask, roi_pts = get_road_roi_mask(frame.shape)
        masked_frame = cv2.bitwise_and(enhanced_frame, enhanced_frame, mask=roi_mask)

        detections = []
        confidences = []

        if model is not None:
            results = model.predict(masked_frame, conf=conf_thresh, iou=0.45, verbose=False)
            for r in results:
                for box in r.boxes:
                    b = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append((b[0], b[1], b[2], b[3]))
                    confidences.append(conf)

        tracked_objects = tracker.update(detections)

        for i, (obj_id, bbox, ready_to_log) in enumerate(tracked_objects):
            x1, y1, x2, y2 = bbox
            conf = confidences[i] if i < len(confidences) else 0.75
            severity = calculate_severity(bbox, frame_area, conf)

            if ready_to_log and (obj_id not in tracker.logged_objects):
                tracker.logged_objects.add(obj_id)
                db.log_pothole(obj_id, lat, lon, conf, severity, frame, bbox)
                print(f" [CONFIRMED] Pothole #{obj_id} | GPS: ({lat:.5f}, {lon:.5f}) | Severity: {severity}")

            color = (0, 0, 255) if severity == "Severe" else ((0, 165, 255) if severity == "Moderate" else (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"#{obj_id} {severity} {conf*100:.0f}%", (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.polylines(frame, [roi_pts], isClosed=True, color=(255, 120, 0), thickness=1)

        cv2.rectangle(frame, (10, 10), (330, 95), (0, 0, 0), -1)
        cv2.putText(frame, f"GPS: {lat:.5f}, {lon:.5f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Speed: {speed} km/h | Mode: {gps_mode}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Verified Potholes: {len(tracker.logged_objects)}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("High-Accuracy Pothole Detection & GPS", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    gps.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="demo_road.mp4")
    parser.add_argument("--gps", type=str, default="simulated", choices=["simulated", "serial"])
    parser.add_argument("--port", type=str, default="COM3")
    parser.add_argument("--model", type=str, default="weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.48)
    args = parser.parse_args()

    run_detection(video_source=args.source, gps_mode=args.gps, gps_port=args.port, model_path=args.model, conf_thresh=args.conf)
