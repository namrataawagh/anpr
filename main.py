"""
main.py — ANPR pipeline entry point.

Pipeline overview
-----------------
1. Vehicle detection  : YOLOv8n (COCO) detects cars, trucks, buses, motorbikes.
2. Vehicle tracking   : SORT assigns a persistent car_id across frames.
3. Plate detection    : Custom YOLOv8 model detects license plate regions.
4. Plate assignment   : Each plate is matched to its containing vehicle.
5. Preprocessing      : Grayscale + Otsu-style inverse binary threshold.
6. OCR                : EasyOCR extracts text; util.py validates + corrects it.
7. CSV export         : Only rows with valid, format-compliant plate text are saved.

Why track vehicles, not plates?
--------------------------------
Vehicle detections (COCO model) are far more stable than plate detections.
By anchoring every plate reading to a stable car_id we can aggregate results
across all frames a car appears in and later pick the best-confidence read,
which significantly reduces single-frame OCR errors.

Usage
-----
    python main.py [--video sample.mp4] [--coco-model yolov8n.pt]
                   [--plate-model ./models/license_plate_detector.pt]
                   [--output ./test.csv] [--max-frames N] [--debug]
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


from util import get_car, read_license_plate, write_csv

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorbike, bus, truck
DEFAULT_VIDEO    = "./sample.mp4"
DEFAULT_COCO     = "yolov8n.pt"
DEFAULT_PLATE    = "./models/license_plate_detector.pt"
DEFAULT_OUTPUT   = "./test.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Indian ANPR — detection + tracking + OCR")
    p.add_argument("--video",       default=DEFAULT_VIDEO,  help="Input video path")
    p.add_argument("--coco-model",  default=DEFAULT_COCO,   help="YOLOv8 COCO model (auto-downloaded)")
    p.add_argument("--plate-model", default=DEFAULT_PLATE,  help="Custom license-plate detector .pt")
    p.add_argument("--output",      default=DEFAULT_OUTPUT, help="Output CSV path")
    p.add_argument("--max-frames",  type=int, default=None, help="Stop after N frames (testing)")
    p.add_argument("--debug",       action="store_true",    help="Show crops in an OpenCV window")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    # SORT must be cloned into ./sort so this import path works.
    try:
        from sort.sort import Sort
    except (ModuleNotFoundError, ImportError):
        sys.exit(
            "[ERROR] SORT tracker not found.\n"
            "Clone the repo and rename the directory:\n"
            "  git clone https://github.com/abewley/sort.git sort\n"
            "Then re-run this script."
        )


    # -- Guard: plate model must exist (user-supplied) ----------------------
    if not os.path.isfile(args.plate_model):
        sys.exit(
            f"[ERROR] License-plate model not found: {args.plate_model}\n"
            "Please supply or train a YOLOv8 model on an Indian license-plate\n"
            "dataset (e.g. from Roboflow) and place it at that path.\n"
            "See README.md for instructions."
        )

    # -- Guard: input video must exist -------------------------------------
    if not os.path.isfile(args.video):
        sys.exit(f"[ERROR] Input video not found: {args.video}")

    # -- Load models -------------------------------------------------------
    print("[INFO] Loading COCO vehicle detector …")
    coco_model = YOLO(args.coco_model)

    print(f"[INFO] Loading license-plate detector from {args.plate_model} …")
    license_plate_detector = YOLO(args.plate_model)

    # SORT tracker — default hyperparameters work well for highway footage.
    mot_tracker = Sort()

    # -- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {total_frames} frames  |  processing up to "
          f"{'all' if args.max_frames is None else args.max_frames} frames")

    results: dict = {}
    frame_nmr: int = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        if args.max_frames is not None and frame_nmr >= args.max_frames:
            break

        results[frame_nmr] = {}

        # ----------------------------------------------------------------
        # 1. Vehicle detection
        # ----------------------------------------------------------------
        detections_ = []
        coco_preds = coco_model(frame)[0]
        for box in coco_preds.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) in VEHICLE_CLASSES:
                detections_.append([x1, y1, x2, y2, score])
                # Note: SORT does not use class_id; drop it here.

        # ----------------------------------------------------------------
        # 2. Vehicle tracking
        # ----------------------------------------------------------------
        if detections_:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            # SORT requires an update call every frame to maintain track state.
            track_ids = mot_tracker.update(np.empty((0, 5)))

        # ----------------------------------------------------------------
        # 3. Plate detection
        # ----------------------------------------------------------------
        license_plates = license_plate_detector(frame, conf=0.15, verbose=False)[0]
        for lp_box in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = lp_box

            # Guard: skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            license_plate = (x1, y1, x2, y2, score, class_id)

            # ----------------------------------------------------------------
            # 4. Assign plate to vehicle
            # ----------------------------------------------------------------
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id == -1:
                continue  # plate not inside any tracked vehicle

            # ----------------------------------------------------------------
            # 5. Crop and preprocess
            # ----------------------------------------------------------------
            ih, iw = frame.shape[:2]

            # Clamp coordinates to frame boundaries
            lx1 = max(0, int(x1))
            ly1 = max(0, int(y1))
            lx2 = min(iw, int(x2))
            ly2 = min(ih, int(y2))

            if lx2 <= lx1 or ly2 <= ly1:
                continue  # zero-size crop after clamping

            license_plate_crop = frame[ly1:ly2, lx1:lx2]

            # Convert to grayscale, then apply inverse binary threshold.
            # The inverse threshold makes the plate characters white on black,
            # which EasyOCR handles well.  Threshold value 64 is intentionally
            # low to preserve dark ink on light plates and light ink on dark plates.
            gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                gray, 64, 255, cv2.THRESH_BINARY_INV
            )

            if args.debug:
                cv2.imshow("Original crop", license_plate_crop)
                cv2.imshow("Thresholded crop", license_plate_crop_thresh)
                cv2.waitKey(1)

            # ----------------------------------------------------------------
            # 6. OCR
            # ----------------------------------------------------------------
            license_plate_text, text_score = read_license_plate(license_plate_crop_thresh)

            if args.debug:
                if license_plate_text is not None:
                    print(f"[DEBUG] Frame {frame_nmr} car {car_id}: "
                          f"plate='{license_plate_text}' score={text_score:.2f}")
                else:
                    # Show what OCR actually read (before validation rejected it)
                    import easyocr as _easyocr
                    _raw = util.reader.readtext(license_plate_crop_thresh)
                    _raw_text = " | ".join(
                        f"'{t}'({s:.2f})" for _, t, s in _raw
                    ) if _raw else "(nothing)"
                    print(f"[DEBUG] Frame {frame_nmr} car {car_id}: "
                          f"plate detected, OCR read: {_raw_text} → rejected by validator")

            # ----------------------------------------------------------------
            # 7. Store result (only if OCR returned a valid plate)
            # ----------------------------------------------------------------
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    "car": {"bbox": [xcar1, ycar1, xcar2, ycar2]},
                    "license_plate": {
                        "bbox":       [x1, y1, x2, y2],
                        "text":       license_plate_text,
                        "bbox_score": score,
                        "text_score": text_score,
                    },
                }

        if frame_nmr % 50 == 0:
            print(f"[INFO] Processed frame {frame_nmr}/{total_frames}")

    # -- Teardown ----------------------------------------------------------
    cap.release()
    if args.debug:
        cv2.destroyAllWindows()

    # -- Write CSV ---------------------------------------------------------
    write_csv(results, args.output)
    print(f"[INFO] Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
