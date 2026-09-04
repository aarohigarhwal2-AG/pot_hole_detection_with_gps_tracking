"""
detect_and_geotag.py
---------------------
Run the trained pothole detector on a dashcam video, and attach a real-world
GPS coordinate to every detected pothole using a synced GPS log.

Two modes:
  1. GPS-synced video (recommended for the real bus deployment):
       python detect_and_geotag.py \
           --weights runs/pothole/bharat_pothole_v1/weights/best.pt \
           --video dashcam_clip.mp4 \
           --gps_file gps_log.gpx --gps_format gpx \
           --video_start "2026-09-03 08:15:00" \
           --out potholes.csv

  2. No GPS log available yet (e.g. testing on a random dashcam clip / the
     BharatPothole sample videos): still runs detection and saves annotated
     video + a CSV, just without lat/lon (marked as None), so you can wire
     GPS in later without changing the detection pipeline.
       python detect_and_geotag.py \
           --weights best.pt --video clip.mp4 --out potholes.csv

Output:
  - `potholes.csv`: one row per detected pothole event with frame number,
    video timestamp, confidence, bbox, and lat/lon (if GPS provided).
  - `potholes.geojson`: same detections as a GeoJSON FeatureCollection, so
    it can be dropped straight into Google My Maps / QGIS / a govt GIS
    portal for pinpointing on a real map.
  - an annotated output video with boxes (optional, --save_video).

De-duplication: a physical pothole is usually seen across many consecutive
frames as the bus approaches it. This script clusters detections that are
close in time (and space, if GPS is available) into a single reported
pothole, so you don't get 30 rows for the same pothole.
"""

import os
import json
import exifread
import pandas as pd
from ultralytics import YOLO


# Convert EXIF Degrees/Minutes/Seconds format to Decimal degrees
def convert_to_decimal(exif_ratio, ref):
    degrees = float(exif_ratio.values[0].num) / float(exif_ratio.values[0].den)
    minutes = float(exif_ratio.values[1].num) / float(exif_ratio.values[1].den)
    seconds = float(exif_ratio.values[2].num) / float(exif_ratio.values[2].den)

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal


# Extract GPS coordinates from image metadata
def get_image_gps(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
                lat = convert_to_decimal(tags['GPS GPSLatitude'], tags['GPS GPSLatitudeRef'].values)
                lon = convert_to_decimal(tags['GPS GPSLongitude'], tags['GPS GPSLongitudeRef'].values)
                return lat, lon
    except Exception as e:
        print(f"Error reading EXIF from {image_path}: {e}")
    return None, None


def main():
    # Path to your trained weights
    model = YOLO(r"weights\best.pt")
    image_folder = "test_photos"
    public_folder = "public"

    os.makedirs(public_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    geotagged_potholes = []

    print(f"Scanning images in '{image_folder}'...")
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)

            # Lowered confidence threshold and enabled logging
            # Force high-resolution inference and lower confidence
            results = model.predict(img_path, conf=0.10, imgsz=1280, verbose=False)
            detected_count = len(results[0].boxes)

            lat, lon = get_image_gps(img_path)
            gps_status = f"({lat:.4f}, {lon:.4f})" if (lat and lon) else "NO GPS FOUND"

            print(f"-> {filename}: {detected_count} pothole(s) detected | GPS: {gps_status}")

            if detected_count > 0:
                if lat and lon:
                    output_path = os.path.join(public_folder, filename)
                    results[0].save(output_path)

                    geotagged_potholes.append({
                        "Image": filename,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Confidence": round(float(results[0].boxes.conf[0]), 3)
                    })
    # Export to JSON for Leaflet frontend
    json_path = os.path.join(public_folder, "map_data.json")
    with open(json_path, "w") as f:
        json.dump(geotagged_potholes, f, indent=4)

    print(f"\nCompleted: {len(geotagged_potholes)} geotagged potholes logged to '{json_path}'.")


if __name__ == "__main__":
    main()