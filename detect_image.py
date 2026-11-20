"""
detect_image.py — Run the ANPR pipeline on a single image (photo).

Usage
-----
    python detect_image.py --image car.jpg
    python detect_image.py --image car.jpg --plate-model models/license_plate_detector.pt
    python detect_image.py --image car.jpg --output result.jpg --show

What it does
------------
1. Detects vehicles (COCO YOLOv8n) in the image.
2. Detects license plate regions (custom plate model).
3. Assigns each plate to its containing vehicle.
4. Crops and preprocesses each plate.
5. Runs EasyOCR and validates against Indian plate formats.
6. Annotates the image with bounding boxes, tracking-style IDs, and plate text.
7. Saves the annotated image and prints results to stdout.

Supports both individual car photos and multi-vehicle scene photos.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

import util
from util import read_license_plate, get_car

VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car, motorbike, bus, truck
DEFAULT_PLATE   = "./models/license_plate_detector.pt"

# Corner rectangle drawing (matches visualize.py style)
def _draw_corner_rect(img, x1, y1, x2, y2, color, thickness=3, r=20):
    for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
        dx = 1 if px == x1 else -1
        dy = 1 if py == y1 else -1
        cv2.line(img, (px, py), (px + dx * r, py),          color, thickness)
        cv2.line(img, (px, py), (px,          py + dy * r), color, thickness)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Indian ANPR — single image / photo detection"
    )
    p.add_argument("--image",        required=True,          help="Input image path (jpg/png/etc.)")
    p.add_argument("--coco-model",   default="yolov8n.pt",   help="YOLOv8 COCO model (auto-downloaded)")
    p.add_argument("--plate-model",  default=DEFAULT_PLATE,  help="License-plate detector .pt")
    p.add_argument("--output",       default=None,           help="Annotated output image path (optional)")
    p.add_argument("--show",         action="store_true",    help="Display result in a window")
    p.add_argument("--conf",         type=float, default=0.15, help="Plate detector confidence threshold")
    p.add_argument("--no-vehicle",   action="store_true",
                   help="Skip vehicle detection — use the entire image as one 'vehicle' region. "
                        "Useful for close-up plate photos.")
    return p.parse_args()


def annotate(
    image: np.ndarray,
    vehicles: list[tuple],
    plates: list[dict],
) -> np.ndarray:
    """Draw vehicle boxes and plate banners onto a copy of *image*."""
    out = image.copy()
    ih, iw = out.shape[:2]

    # --- Vehicle boxes ---
    for i, (vx1, vy1, vx2, vy2) in enumerate(vehicles):
        _draw_corner_rect(out, int(vx1), int(vy1), int(vx2), int(vy2),
                          color=(0, 200, 255), thickness=3)
        label = f"Vehicle {i+1}"
        cv2.putText(out, label, (int(vx1), int(vy1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # --- Plate results ---
    for p in plates:
        px1, py1, px2, py2 = int(p["x1"]), int(p["y1"]), int(p["x2"]), int(p["y2"])
        text   = p["text"]
        score  = p["score"]
        crop   = p["crop"]

        # Red plate rectangle
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 0, 220), 2)

        # -- White banner above plate --
        SCALE   = 1.5
        PADDING = 6
        FONT    = cv2.FONT_HERSHEY_SIMPLEX
        FSCALE  = 0.55
        THICK   = 1

        (tw, th), baseline = cv2.getTextSize(text, FONT, FSCALE, THICK)

        # Resize crop to banner height
        crop_h = th + baseline + PADDING * 2
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            aspect = crop.shape[1] / crop.shape[0]
            crop_w = int(crop_h * aspect)
            crop_resized = cv2.resize(crop, (crop_w, crop_h))
            if len(crop_resized.shape) == 2:
                crop_resized = cv2.cvtColor(crop_resized, cv2.COLOR_GRAY2BGR)
        else:
            crop_w, crop_resized = 0, None

        gap        = 4
        banner_w   = min(iw, crop_w + gap + tw + PADDING * 2)
        banner_h   = crop_h

        # Position banner above plate
        bx1 = max(0, min(px1, iw - banner_w))
        by2 = max(0, py1 - 2)
        by1 = max(0, by2 - banner_h)

        if by1 < by2 and bx1 + banner_w <= iw:
            # White background
            cv2.rectangle(out, (bx1, by1), (bx1 + banner_w, by2), (255, 255, 255), -1)
            # Paste crop
            if crop_resized is not None and crop_w > 0:
                cx_end = min(bx1 + crop_w, iw)
                actual_cw = cx_end - bx1
                out[by1:by2, bx1:cx_end] = crop_resized[:, :actual_cw]
            # Plate text
            tx = bx1 + crop_w + gap + PADDING
            ty = by1 + th + PADDING
            cv2.putText(out, text, (tx, ty), FONT, FSCALE, (20, 20, 20), THICK)
            # Score badge
            score_txt = f"{score:.0%}"
            cv2.putText(out, score_txt, (tx, ty + baseline + 2),
                        FONT, 0.4, (100, 100, 100), 1)

    return out


def main() -> None:
    args = parse_args()

    # --- Guards ---
    if not os.path.isfile(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")
    if not os.path.isfile(args.plate_model):
        sys.exit(
            f"[ERROR] Plate model not found: {args.plate_model}\n"
            "Download or train a YOLOv8 plate detector and place it at that path."
        )

    # --- Load image ---
    image = cv2.imread(args.image)
    if image is None:
        sys.exit(f"[ERROR] Could not read image: {args.image}")
    ih, iw = image.shape[:2]
    print(f"[INFO] Image: {iw}×{ih}  |  {args.image}")

    # --- Load models ---
    print("[INFO] Loading COCO vehicle detector …")
    coco_model = YOLO(args.coco_model)

    print(f"[INFO] Loading plate detector from {args.plate_model} …")
    plate_model = YOLO(args.plate_model)

    # -------------------------------------------------------------------
    # 1. Vehicle detection
    # -------------------------------------------------------------------
    vehicles: list[tuple] = []
    track_ids_arr: np.ndarray

    if args.no_vehicle:
        # Treat entire image as one vehicle region
        vehicles = [(0, 0, iw, ih)]
        track_ids_arr = np.array([[0, 0, iw, ih, 1]], dtype=float)
        print("[INFO] --no-vehicle: treating full image as one vehicle region")
    else:
        coco_preds = coco_model(image, verbose=False)[0]
        detections = []
        for box in coco_preds.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) in VEHICLE_CLASSES:
                detections.append((x1, y1, x2, y2, score))
                vehicles.append((x1, y1, x2, y2))

        if not detections:
            print("[WARN] No vehicles detected. Trying --no-vehicle mode automatically …")
            vehicles = [(0, 0, iw, ih)]
            track_ids_arr = np.array([[0, 0, iw, ih, 1]], dtype=float)
        else:
            # Build a fake track_ids array (same format SORT produces: x1,y1,x2,y2,id)
            track_ids_arr = np.array(
                [[x1, y1, x2, y2, i + 1]
                 for i, (x1, y1, x2, y2, _) in enumerate(detections)],
                dtype=float,
            )
            print(f"[INFO] Detected {len(vehicles)} vehicle(s)")

    # -------------------------------------------------------------------
    # 2. Plate detection
    # -------------------------------------------------------------------
    plate_preds = plate_model(image, conf=args.conf, verbose=False)[0]
    raw_plates  = plate_preds.boxes.data.tolist()
    print(f"[INFO] Plate model found {len(raw_plates)} candidate region(s) "
          f"at conf≥{args.conf}")

    # -------------------------------------------------------------------
    # 3. Assign plates → vehicles, crop, preprocess, OCR
    # -------------------------------------------------------------------
    results: list[dict] = []

    for lp_box in raw_plates:
        x1, y1, x2, y2, lp_score, _ = lp_box

        if x2 <= x1 or y2 <= y1:
            continue

        # Match to vehicle
        lp_tuple = (x1, y1, x2, y2, lp_score, 0)
        _, _, _, _, car_id = get_car(lp_tuple, track_ids_arr)
        if car_id == -1:
            # Not inside any detected vehicle — if no-vehicle mode, accept anyway
            if not args.no_vehicle and len(vehicles) > 0:
                # still try — assign to nearest vehicle heuristically
                car_id = 1
            else:
                car_id = 1

        # Crop
        lx1, ly1 = max(0, int(x1)), max(0, int(y1))
        lx2, ly2 = min(iw, int(x2)), min(ih, int(y2))
        if lx2 <= lx1 or ly2 <= ly1:
            continue

        crop_bgr = image[ly1:ly2, lx1:lx2]
        gray     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        # OCR
        text, text_score = read_license_plate(thresh)

        raw_ocr = util.reader.readtext(thresh)
        raw_txt = " | ".join(f"'{t}'({s:.2f})" for _, t, s in raw_ocr) if raw_ocr else "(nothing)"

        if text is not None:
            results.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "text": text, "score": text_score,
                "plate_score": lp_score,
                "crop": crop_bgr,
            })
            print(f"  ✅  Plate: {text}  (OCR score: {text_score:.2%}, "
                  f"detector conf: {lp_score:.2%})")
        else:
            print(f"  ❌  Region conf={lp_score:.2f}: OCR read {raw_txt} "
                  f"→ failed Indian format validation")

    # -------------------------------------------------------------------
    # 4. Annotate and output
    # -------------------------------------------------------------------
    if results:
        print(f"\n[INFO] {len(results)} valid Indian plate(s) found:")
        for r in results:
            print(f"       {r['text']}  (score {r['score']:.2%})")
    else:
        print("\n[INFO] No valid Indian-format plates found in this image.")
        print("       This may be a non-Indian plate, or the image quality")
        print("       may be too low for OCR. Try --no-vehicle for close-up shots.")

    annotated = annotate(image, vehicles, results)

    # Determine output path
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.image)
        out_path  = base + "_anpr" + (ext if ext else ".jpg")

    cv2.imwrite(out_path, annotated)
    print(f"\n[INFO] Annotated image saved → {out_path}")

    if args.show:
        cv2.imshow("ANPR Result", annotated)
        print("[INFO] Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
