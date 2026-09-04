"""
convert_to_yolo.py
-------------------
Optional helper: only needed if BharatPothole is provided in COCO JSON
or Pascal VOC XML format instead of native YOLO .txt labels.

Usage:
    # COCO -> YOLO
    python convert_to_yolo.py --format coco \
        --annotations bharat_pothole/train/_annotations.coco.json \
        --images bharat_pothole/train/images \
        --out_labels bharat_pothole/train/labels

    # Pascal VOC -> YOLO
    python convert_to_yolo.py --format voc \
        --annotations bharat_pothole/train/annotations_xml \
        --images bharat_pothole/train/images \
        --out_labels bharat_pothole/train/labels
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET

from PIL import Image


def coco_to_yolo(annotations_path, out_labels_dir):
    os.makedirs(out_labels_dir, exist_ok=True)
    with open(annotations_path, "r") as f:
        coco = json.load(f)

    # Map category_id -> contiguous class index (0-based, sorted by id)
    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    print("Class mapping (put this in data.yaml 'names:'):")
    for c in cats:
        print(f"  {cat_id_to_idx[c['id']]}: {c['name']}")

    images_by_id = {img["id"]: img for img in coco["images"]}
    labels_by_image = {}

    for ann in coco["annotations"]:
        img = images_by_id[ann["image_id"]]
        w, h = img["width"], img["height"]
        x, y, bw, bh = ann["bbox"]  # COCO bbox = [x_min, y_min, width, height]

        cx = (x + bw / 2) / w
        cy = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h
        cls = cat_id_to_idx[ann["category_id"]]

        line = f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        labels_by_image.setdefault(img["file_name"], []).append(line)

    for file_name, lines in labels_by_image.items():
        stem = os.path.splitext(os.path.basename(file_name))[0]
        with open(os.path.join(out_labels_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(labels_by_image)} YOLO label files to {out_labels_dir}")


def voc_to_yolo(annotations_dir, images_dir, out_labels_dir, class_list=None):
    os.makedirs(out_labels_dir, exist_ok=True)
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

    # Discover classes if not provided
    if class_list is None:
        classes = set()
        for xf in xml_files:
            tree = ET.parse(os.path.join(annotations_dir, xf))
            for obj in tree.getroot().findall("object"):
                classes.add(obj.find("name").text)
        class_list = sorted(classes)
        print("Discovered classes (put this in data.yaml 'names:'):")
        for i, c in enumerate(class_list):
            print(f"  {i}: {c}")

    class_to_idx = {c: i for i, c in enumerate(class_list)}

    for xf in xml_files:
        tree = ET.parse(os.path.join(annotations_dir, xf))
        root = tree.getroot()
        stem = os.path.splitext(xf)[0]

        size = root.find("size")
        if size is not None:
            w = float(size.find("width").text)
            h = float(size.find("height").text)
        else:
            img_path = os.path.join(images_dir, stem + ".jpg")
            with Image.open(img_path) as im:
                w, h = im.size

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in class_to_idx:
                continue
            cls = class_to_idx[cls_name]
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            nw = (xmax - xmin) / w
            nh = (ymax - ymin) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(os.path.join(out_labels_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(xml_files)} YOLO label files to {out_labels_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["coco", "voc"], required=True)
    ap.add_argument("--annotations", required=True,
                     help="COCO json file OR VOC xml folder")
    ap.add_argument("--images", required=True, help="images folder")
    ap.add_argument("--out_labels", required=True, help="output YOLO labels folder")
    args = ap.parse_args()

    if args.format == "coco":
        coco_to_yolo(args.annotations, args.out_labels)
    else:
        voc_to_yolo(args.annotations, args.images, args.out_labels)
