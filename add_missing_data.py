"""
add_missing_data.py — Interpolate bounding-box gaps in the ANPR CSV.

Why interpolation?
------------------
SORT assigns a persistent car_id across all frames a vehicle is visible.
However, the license plate text is only stored for frames where:
  1. The plate detector fired.
  2. OCR produced a string that passed format validation.

This leaves gaps in the CSV.  For smooth visualisation we linearly interpolate
the car bbox and plate bbox for every missing frame within a car's track span.
The license_number column in interpolated rows either inherits the best-scoring
text for that car or is left empty — here we copy the car's best reading so the
overlay always shows a plate string.

Usage
-----
    python add_missing_data.py [--input test.csv] [--output test_interpolated.csv]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_INPUT  = "./test.csv"
DEFAULT_OUTPUT = "./test_interpolated.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interpolate missing ANPR bounding boxes")
    p.add_argument("--input",  default=DEFAULT_INPUT,  help="Input CSV (from main.py)")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output interpolated CSV")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_bbox(bbox_str: str) -> list[float]:
    """
    Parse a bounding-box string of the form "[x1 y1 x2 y2]" into a list of
    four floats.  Accepts both space- and comma-separated values.
    """
    nums = re.findall(r"[-\d.]+", str(bbox_str))
    if len(nums) != 4:
        raise ValueError(f"Cannot parse bbox string: {bbox_str!r}")
    return [float(n) for n in nums]


def _bbox_to_str(coords: list[float]) -> str:
    return "[{} {} {} {}]".format(*[int(round(c)) for c in coords])


def interpolate_bboxes(
    frames: np.ndarray,
    bboxes: np.ndarray,
    all_frames: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate *bboxes* (shape N×4) at *all_frames* given known
    readings at *frames*.

    Parameters
    ----------
    frames     : 1-D int array of frame numbers where bbox is known
    bboxes     : 2-D float array (N, 4)
    all_frames : 1-D int array of every frame number to fill

    Returns
    -------
    np.ndarray of shape (len(all_frames), 4)

    Notes
    -----
    When only a single frame is known (len(frames) == 1), linear interp1d
    raises ValueError because it needs >= 2 points.  In that case we simply
    broadcast the single known bbox across all requested frames.
    """
    interpolated = np.zeros((len(all_frames), 4), dtype=float)
    if len(frames) == 1:
        # Single known frame: broadcast the bbox to every output frame
        for col in range(4):
            interpolated[:, col] = bboxes[0, col]
        return interpolated
    for col in range(4):
        fn = interp1d(
            frames,
            bboxes[:, col],
            kind="linear",
            fill_value="extrapolate",
        )
        interpolated[:, col] = fn(all_frames)
    return interpolated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        sys.exit(f"[ERROR] Input CSV not found: {args.input}\nRun main.py first.")

    if df.empty:
        sys.exit("[ERROR] Input CSV is empty — no valid plates were detected.")

    # Determine the full frame range from the data
    all_frame_numbers = np.arange(df["frame_nmr"].min(), df["frame_nmr"].max() + 1)

    # Identify the car with the best plate score for the license_number fallback
    best_scores: dict[int, str] = {}
    for car_id, grp in df.groupby("car_id"):
        valid = grp[grp["license_number_score"].notna() & (grp["license_number_score"] != 0)]
        if not valid.empty:
            best_row = valid.loc[valid["license_number_score"].idxmax()]
            best_scores[int(car_id)] = str(best_row["license_number"])
        else:
            best_scores[int(car_id)] = ""

    rows: list[dict] = []

    for car_id, grp in df.groupby("car_id"):
        car_id = int(car_id)
        grp = grp.sort_values("frame_nmr").reset_index(drop=True)

        known_frames = grp["frame_nmr"].values.astype(int)
        frame_min    = known_frames.min()
        frame_max    = known_frames.max()
        fill_frames  = np.arange(frame_min, frame_max + 1)

        # Parse car bboxes
        try:
            car_bboxes = np.array([_parse_bbox(b) for b in grp["car_bbox"]], dtype=float)
        except ValueError as e:
            print(f"[WARN] car_id {car_id}: {e} — skipping interpolation for this car.")
            for _, row in grp.iterrows():
                rows.append(row.to_dict())
            continue

        # Parse plate bboxes
        try:
            lp_bboxes = np.array([_parse_bbox(b) for b in grp["license_plate_bbox"]], dtype=float)
        except ValueError as e:
            print(f"[WARN] car_id {car_id}: {e} — skipping plate bbox interpolation.")
            lp_bboxes = car_bboxes.copy()  # fallback: use car bbox

        # Interpolate
        car_interp = interpolate_bboxes(known_frames, car_bboxes, fill_frames)
        lp_interp  = interpolate_bboxes(known_frames, lp_bboxes,  fill_frames)

        # Build lookup for known rows so we can copy exact OCR values
        known_lookup: dict[int, pd.Series] = {
            int(r["frame_nmr"]): r for _, r in grp.iterrows()
        }

        best_text  = best_scores.get(car_id, "")
        best_score_val = grp["license_number_score"].max() if not grp["license_number_score"].isna().all() else 0.0

        for i, fn in enumerate(fill_frames):
            fn = int(fn)
            if fn in known_lookup:
                rows.append(known_lookup[fn].to_dict())
            else:
                rows.append(
                    {
                        "frame_nmr":               fn,
                        "car_id":                  car_id,
                        "car_bbox":                _bbox_to_str(car_interp[i].tolist()),
                        "license_plate_bbox":      _bbox_to_str(lp_interp[i].tolist()),
                        "license_plate_bbox_score": 0,
                        "license_number":          best_text,
                        "license_number_score":    best_score_val,
                    }
                )

    out_df = pd.DataFrame(rows, columns=df.columns).sort_values(
        ["frame_nmr", "car_id"]
    )
    out_df.to_csv(args.output, index=False)
    print(f"[INFO] Interpolated data saved to {args.output}  ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
