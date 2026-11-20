"""
visualize.py — Render an annotated output video from the interpolated ANPR CSV.

Annotation scheme
-----------------
* Corner-style rectangle around each tracked vehicle (thick lines at the 4
  corners, not a full box) — less visually cluttered than a solid rectangle.
* Thin red rectangle around the detected license plate region.
* Above the car: a white banner containing the cropped plate image (extracted
  from the video at the frame where the best OCR score was achieved) and the
  plate text centered below it.

Why use the best-score plate per car_id?
-----------------------------------------
OCR on individual frames is noisy.  Across a car's full track there may be
dozens of plate detections.  We pick the frame with the highest
license_number_score and use that plate crop and that text string as the
canonical result for the entire video.  This yields a stable, clean label
even when most individual reads are partially wrong.

Usage
-----
    python visualize.py [--csv test_interpolated.csv]
                        [--video sample.mp4]
                        [--output out.mp4]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLATE_CROP_HEIGHT   = 400       # px — fixed height for the plate overlay image
CORNER_LENGTH_FRAC  = 0.15      # fraction of box side used for corner arms
CORNER_THICKNESS    = 6         # px
PLATE_RECT_THICKNESS= 2         # px — thin red rectangle around the plate
FONT                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE          = 1.2
FONT_THICKNESS      = 2
TEXT_PADDING        = 8         # px above/below text inside banner


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render annotated ANPR output video")
    p.add_argument("--csv",    default="./test_interpolated.csv", help="Interpolated CSV")
    p.add_argument("--video",  default="./sample.mp4",            help="Original input video")
    p.add_argument("--output", default="./out.mp4",               help="Output annotated video")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bbox(bbox_str: str) -> tuple[int, int, int, int]:
    nums = re.findall(r"[-\d.]+", str(bbox_str))
    if len(nums) != 4:
        raise ValueError(f"Cannot parse bbox: {bbox_str!r}")
    return tuple(int(round(float(n))) for n in nums)  # type: ignore[return-value]


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def _draw_corner_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
    thickness: int,
    length_frac: float,
) -> None:
    """
    Draw a corner-style rectangle (four L-shaped corners) instead of a full
    bounding box.  This keeps the annotation unobtrusive while still clearly
    marking the region of interest.
    """
    w = x2 - x1
    h = y2 - y1
    lx = max(10, int(w * length_frac))
    ly = max(10, int(h * length_frac))

    # Top-left
    cv2.line(img, (x1, y1), (x1 + lx, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + ly), color, thickness)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - lx, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + ly), color, thickness)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + lx, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - ly), color, thickness)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - lx, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - ly), color, thickness)


def _draw_banner(
    frame: np.ndarray,
    car_x1: int, car_y1: int, car_x2: int,
    plate_crop: np.ndarray,
    plate_text: str,
) -> None:
    """
    Draw a white banner above the car bounding box containing:
      * the best-resolution plate crop (top)
      * the plate text centered below the image (bottom)
    """
    ih, iw = frame.shape[:2]

    # --- Resize plate crop to fixed height --------------------------------
    ph, pw = plate_crop.shape[:2]
    if ph == 0 or pw == 0:
        return
    aspect = pw / ph
    new_h  = PLATE_CROP_HEIGHT
    new_w  = int(new_h * aspect)
    plate_resized = cv2.resize(plate_crop, (new_w, new_h))

    # --- Measure text -----------------------------------------------------
    (tw, th), baseline = cv2.getTextSize(plate_text, FONT, FONT_SCALE, FONT_THICKNESS)
    text_block_h = th + baseline + 2 * TEXT_PADDING

    banner_w = max(new_w, tw + 2 * TEXT_PADDING)
    banner_w = min(banner_w, iw)   # never wider than the frame
    banner_h = new_h + text_block_h
    banner_h = min(banner_h, ih)   # never taller than the frame

    # --- Position: centred horizontally above car box ---------------------
    car_cx = (car_x1 + car_x2) // 2
    bx1    = _clamp(car_cx - banner_w // 2, 0, max(0, iw - banner_w))
    bx2    = bx1 + banner_w
    by2    = _clamp(car_y1 - 10, banner_h, ih)  # 10 px gap below car top
    by1    = by2 - banner_h

    if by1 < 0:
        by1 = 0
        by2 = banner_h

    # --- Draw white background --------------------------------------------
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), -1)

    # --- Paste plate image (centred) --------------------------------------
    img_x = bx1 + (banner_w - new_w) // 2
    img_y = by1
    img_x = _clamp(img_x, 0, iw - new_w)
    img_y = _clamp(img_y, 0, ih - new_h)

    dest = frame[img_y: img_y + new_h, img_x: img_x + new_w]
    if dest.shape[:2] == plate_resized.shape[:2]:
        frame[img_y: img_y + new_h, img_x: img_x + new_w] = plate_resized

    # --- Draw text (centred) below image ----------------------------------
    tx = bx1 + (banner_w - tw) // 2
    ty = by1 + new_h + TEXT_PADDING + th
    tx = _clamp(tx, 0, iw - tw)
    ty = _clamp(ty, th, ih)
    cv2.putText(frame, plate_text, (tx, ty), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Load CSV ----------------------------------------------------------
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        sys.exit(f"[ERROR] CSV not found: {args.csv}\nRun add_missing_data.py first.")

    if df.empty:
        sys.exit("[ERROR] Interpolated CSV is empty.")

    # -- Open video --------------------------------------------------------
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -- Identify best plate text + frame per car_id -----------------------
    # We pick the row with the highest license_number_score per car.
    # Using that frame we crop the plate image directly from the source video
    # so the overlay reflects real pixel data, not an interpolated box.
    best: dict[int, dict] = {}
    for car_id, grp in df.groupby("car_id"):
        valid = grp[grp["license_number_score"].notna() & (grp["license_number_score"] > 0)]
        if valid.empty:
            valid = grp
        best_row = valid.loc[valid["license_number_score"].idxmax()]
        best[int(car_id)] = {
            "frame":  int(best_row["frame_nmr"]),
            "text":   str(best_row["license_number"]),
            "lp_bbox": best_row["license_plate_bbox"],
        }

    # -- Extract best plate crops from the source video --------------------
    plate_crops: dict[int, np.ndarray] = {}
    needed_frames: dict[int, list[int]] = {}  # frame -> [car_id, ...]

    for car_id, info in best.items():
        needed_frames.setdefault(info["frame"], []).append(car_id)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    while needed_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in needed_frames:
            for car_id in needed_frames.pop(frame_idx):
                try:
                    x1, y1, x2, y2 = _parse_bbox(best[car_id]["lp_bbox"])
                    ih, iw = frame.shape[:2]
                    x1 = _clamp(x1, 0, iw - 1)
                    y1 = _clamp(y1, 0, ih - 1)
                    x2 = _clamp(x2, x1 + 1, iw)
                    y2 = _clamp(y2, y1 + 1, ih)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        plate_crops[car_id] = crop
                except (ValueError, Exception) as e:
                    print(f"[WARN] car_id {car_id}: could not extract plate crop: {e}")
        frame_idx += 1

    # -- Set up VideoWriter -----------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        sys.exit(f"[ERROR] Cannot create output video: {args.output}")

    # Pre-index CSV by frame for fast per-frame lookup
    frame_index: dict[int, pd.DataFrame] = {
        fn: grp for fn, grp in df.groupby("frame_nmr")
    }

    # -- Main render loop --------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fn in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        rows = frame_index.get(fn)
        if rows is not None:
            for _, row in rows.iterrows():
                car_id = int(row["car_id"])

                # -- Car corner rectangle ----------------------------------
                # Initialize with safe defaults so _draw_banner can always
                # reference cx1/cy1/cx2 even if the parse try-block fails.
                cx1 = cy1 = cx2 = cy2 = 0
                try:
                    cx1, cy1, cx2, cy2 = _parse_bbox(row["car_bbox"])
                    ih, iw = frame.shape[:2]
                    cx1 = _clamp(cx1, 0, iw)
                    cy1 = _clamp(cy1, 0, ih)
                    cx2 = _clamp(cx2, 0, iw)
                    cy2 = _clamp(cy2, 0, ih)
                    _draw_corner_rect(
                        frame, cx1, cy1, cx2, cy2,
                        color=(0, 255, 0),
                        thickness=CORNER_THICKNESS,
                        length_frac=CORNER_LENGTH_FRAC,
                    )
                except ValueError:
                    pass

                # -- Plate rectangle ---------------------------------------
                try:
                    px1, py1, px2, py2 = _parse_bbox(row["license_plate_bbox"])
                    ih, iw = frame.shape[:2]
                    px1 = _clamp(px1, 0, iw)
                    py1 = _clamp(py1, 0, ih)
                    px2 = _clamp(px2, 0, iw)
                    py2 = _clamp(py2, 0, ih)
                    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), PLATE_RECT_THICKNESS)
                except ValueError:
                    pass

                # -- Plate banner above car --------------------------------
                plate_text = best.get(car_id, {}).get("text", "")
                crop       = plate_crops.get(car_id)
                if crop is not None and plate_text:
                    try:
                        _draw_banner(frame, cx1, cy1, cx2, crop, plate_text)
                    except Exception as e:
                        print(f"[WARN] frame {fn}, car_id {car_id}: banner error: {e}")

        out.write(frame)

        if fn % 100 == 0:
            print(f"[INFO] Rendered frame {fn}/{total}")

    cap.release()
    out.release()
    print(f"[INFO] Output video saved to {args.output}")


if __name__ == "__main__":
    main()
