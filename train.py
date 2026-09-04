"""
train.py
--------
Train a pothole detector on the BharatPothole dataset using YOLOv8.

Usage:
    python train.py --data data.yaml --model yolov8n.pt --epochs 100 --imgsz 640

Notes for a dashcam use-case:
- Start with yolov8n.pt (nano) or yolov8s.pt (small) — these run fast enough
  for near-real-time inference on modest hardware / an on-bus edge device
  (Jetson Nano/Orin, etc). Only go to yolov8m/l if you have a GPU on the bus
  unit and need higher accuracy.
- imgsz=640 is a good default; potholes are often small in dashcam frames,
  so if recall is poor try imgsz=960 or add tiling/slicing at inference time.


import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="the road dataset/BharatPotHole/BharatPotHole/data.yaml", help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt",
                     help="base weights: yolov8n/s/m/l/x.pt")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0, help="GPU id, or 'cpu'")
    ap.add_argument("--project", default="runs/pothole")
    ap.add_argument("--name", default="bharat_pothole_v1")
    ap.add_argument("--patience", type=int, default=20,
                     help="early stopping patience (epochs w/o improvement)")
    args = ap.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device='cpu',
        project=args.project,
        name=args.name,
        patience=args.patience,
        # Augmentations that help with dashcam conditions: motion blur-ish
        # jitter, varied lighting/weather, and scale variance for potholes
        # that appear near vs far from the camera.
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0,
    )

    # Validate on the test split too, if present
    metrics = model.val(data=args.data, split="test", device='cpu')
    print("Test metrics:", metrics.results_dict)

    best_path = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\nBest weights saved at: {best_path}")
    print("Use this path as --weights in detect_and_geotag.py")


if __name__ == "__main__":
    main()
"""
import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="the road dataset/BharatPotHole/BharatPotHole/data.yaml", help="path to data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="base weights: yolov8n/s/m/l/x.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=0, help="GPU id, or 'cpu'")
    ap.add_argument("--project", default="runs/pothole")
    ap.add_argument("--name", default="bharat_pothole_v1")
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=8,  # Lowered from 16 to save memory
        workers=2,  # Limits background dataloader memory duplication on Windows
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5, flipud=0.0,
        mosaic=1.0,
    )

    # Validate on the test split
    metrics = model.val(data=args.data, split="test", device=args.device)
    print("Test metrics:", metrics.results_dict)

    best_path = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\nBest weights saved at: {best_path}")


if __name__ == "__main__":
    main()
