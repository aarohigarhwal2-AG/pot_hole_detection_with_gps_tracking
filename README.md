# Road Pothole Detection & Geotagging (BharatPothole → Bus Dashcam)

Detects potholes in dashcam footage and outputs their real-world GPS
coordinates so a government road-maintenance body can pinpoint and fix them.

## Pipeline

```
BharatPothole dataset ─▶ train.py ─▶ best.pt (trained YOLO model)
                                          │
dashcam video + GPS log ─▶ detect_and_geotag.py ─▶ potholes.csv / potholes.geojson
                                          │
                                    map_visualize.py ─▶ potholes_map.html
```

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Prepare the dataset

Download BharatPothole and arrange it as described in `data.yaml`
(train/valid/test, each with `images/` and `labels/`). If it's not already
in YOLO `.txt` label format, convert it first:

```bash
python convert_to_yolo.py --format coco \
    --annotations bharat_pothole/train/_annotations.coco.json \
    --images bharat_pothole/train/images \
    --out_labels bharat_pothole/train/labels
```

Edit `data.yaml` so `path:` points at your dataset folder.

## 3. Train

```bash
python train.py --data data.yaml --model yolov8n.pt --epochs 100 --imgsz 640
```

- Use `yolov8n.pt` (nano) or `yolov8s.pt` for speed on an edge device (the
  actual bus unit likely won't have a big GPU); go up to `yolov8m.pt` if you
  train/infer on a proper GPU server instead.
- Best weights land at `runs/pothole/bharat_pothole_v1/weights/best.pt`.

## 4. Detect potholes on dashcam video + tag GPS coordinates

For now (before the real bus rig exists), you can still test the detector
on any BharatPothole sample video or your own test clip without GPS —
lat/lon will just come back empty:

```bash
python detect_and_geotag.py --weights best.pt --video test_clip.mp4 \
    --out potholes.csv --save_video annotated.mp4
```

Once you have a GPS log recorded alongside the dashcam video (phone GPS
logger app exporting `.gpx`, or a GPS module logging NMEA/CSV), sync it in:

```bash
python detect_and_geotag.py \
    --weights best.pt \
    --video bus_dashcam.mp4 \
    --gps_file bus_gps_log.gpx --gps_format gpx \
    --video_start "2026-09-03 08:15:00" \
    --out potholes.csv --geojson potholes.geojson --save_video annotated.mp4
```

`--video_start` is the real-world clock time the video recording began —
this is what lets the script match "this pothole appeared at 00:42 into the
video" to "the bus was at lat X, lon Y at 08:15:42".

Nearby repeated detections of the same pothole (seen across many frames as
the bus drives up to it) are automatically merged into a single event.

## 5. Visualize on a map (for the govt / municipal body)

```bash
python map_visualize.py --csv potholes.csv --out potholes_map.html
```

Open `potholes_map.html` in a browser — each red pin is a pothole with its
confidence score and coordinates. `potholes.geojson` can also be imported
directly into Google My Maps, QGIS, or most government GIS portals.

## Getting a GPS log for the real bus deployment

Since the actual dashcam won't have GPS built in, pick one:
1. **Phone app** (e.g. any GPS logger app) running in the bus, exporting a
   `.gpx` file — simplest to set up, no extra hardware.
2. **Cheap GPS module** (e.g. u-blox NEO-6M) wired to a Raspberry Pi/Arduino
   logging NMEA sentences with timestamps to an SD card.
3. If you later upgrade to a combined dashcam+GPS unit, it may embed GPS
   directly in the video (as metadata or a sidecar `.srt`) — in that case
   you can skip `gps_sync.py` and parse that format instead.

Keep the phone/GPS logger's clock and the dashcam's clock reasonably synced
(within a couple of seconds) so `--video_start` lines up correctly.

## Notes / things to tune later

- **Small object size**: potholes far from the camera are tiny in the
  frame. If recall is low, try `--imgsz 960` in training, or run inference
  on cropped/tiled sub-regions of each frame (road surface is usually the
  bottom half of a dashcam frame).
- **False positives from shadows/manholes/wet patches**: add hard-negative
  examples of these to training data if you see this in practice.
- **Real-time on-bus inference**: `yolov8n` at 640px can hit real-time
  speeds on a Jetson Nano/Orin; test actual FPS on your target hardware
  before deciding on model size.
- **De-duplication across multiple bus trips**: if the same pothole is
  reported by multiple trips/buses over time, cluster `potholes.geojson`
  points across trips (e.g. by proximity within ~15-20m) before sending a
  final report to the government, so one physical pothole isn't reported
  as 10 separate ones.
