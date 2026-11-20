"""
util.py — Helper utilities for the Indian ANPR pipeline.

Design notes
------------
* We track *vehicles* (not plates) with SORT because plate detections are
  noisy and intermittent.  By anchoring every plate reading to a stable
  car_id we can later pick the highest-confidence OCR result over all frames
  that car appeared, dramatically reducing single-frame errors.

* The character confusion maps are applied *position-aware*: a digit slot
  that the OCR misread as a letter gets corrected using dict_char_to_int,
  and a letter slot misread as a digit gets corrected using dict_int_to_char.
  This avoids turning a correct '0' in a letter position into an 'O' when it
  was already right, etc.

* The format validator is intentionally kept as a list of compiled regexes so
  that adding a new Indian registration variant (e.g. Army plates AA-00-AA-0000
  or old two-letter district codes) requires only appending a new regex +
  corresponding formatter, with no changes to the rest of the pipeline.
"""

from __future__ import annotations

import os
import re
import csv
from typing import Optional

import cv2
import easyocr
import numpy as np

# ---------------------------------------------------------------------------
# EasyOCR reader — initialised once at module level to avoid reloading the
# model on every call.  Set USE_GPU = True if a CUDA-capable GPU is available.
# ---------------------------------------------------------------------------
USE_GPU: bool = bool(int(os.environ.get("ANPR_GPU", "0")))
reader: easyocr.Reader = easyocr.Reader(["en"], gpu=USE_GPU)

# ---------------------------------------------------------------------------
# Character confusion maps
# These cover the most common OCR substitutions on number-plate fonts.
# ---------------------------------------------------------------------------
dict_char_to_int: dict[str, str] = {
    "O": "0",
    "I": "1",
    "J": "3",
    "A": "4",
    "G": "6",
    "S": "5",
}

dict_int_to_char: dict[str, str] = {
    "0": "O",
    "1": "I",
    "3": "J",
    "4": "A",
    "6": "G",
    "5": "S",
}

# ---------------------------------------------------------------------------
# Indian registration-plate format definitions
#
# Each entry is a dict with:
#   "name"    – human-readable label
#   "regex"   – compiled pattern that the *raw* (already uppercased,
#               spaces-stripped) text must fully match.
#   "formatter" – callable(str) -> str that applies position-aware
#               character correction.
#
# To add a new variant, append a new dict here.  No other code changes needed.
# ---------------------------------------------------------------------------

def _precorrect_standard(text: str) -> str:
    """
    Pre-correct a noisy string that is *approximately* a Standard Indian plate.

    The strategy is a left-to-right greedy scan that uses the expected character
    type at each position to decide which confusion map to apply:
      pos 0-1  -> letter  (state code)
      pos 2-3  -> digit   (RTO, 1 or 2 digits)
      next 1-3 -> letter  (series)
      rest     -> digit   (unique number)

    Guard: only attempt pre-correction when the string length falls within the
    valid range for a Standard Indian plate (6-10 characters), to avoid
    falsely validating plates that are structurally too long or too short.
    """
    # Standard plate: min 2+1+1+1 = 5, max 2+2+3+4 = 11, but realistically 6-10
    if not (5 <= len(text) <= 10):
        return text  # don't attempt correction on clearly wrong-length strings

    out = list(text)
    n   = len(out)

    # State code: positions 0-1 -> letter
    for i in range(min(2, n)):
        out[i] = dict_int_to_char.get(out[i], out[i])

    # RTO code: position 2, optionally 3 -> digit (max 2 digits)
    rto_len = 0
    for i in range(2, min(4, n)):
        corrected = dict_char_to_int.get(out[i], out[i])
        if corrected.isdigit() or out[i].isdigit():
            out[i] = corrected
            rto_len += 1
        else:
            break
    if rto_len == 0:
        out[2] = dict_char_to_int.get(out[2], out[2])
        rto_len = 1

    # Series: next 1-3 chars -> letter.
    # Only consume *natively* alphabetic characters here.  Digits that the
    # confusion map could convert to letters (e.g. '1'->I, '3'->J) are left
    # for the number-correction step, preventing them from being misread as a
    # 3rd series character.
    series_start = 2 + rto_len
    max_series   = min(3, n - series_start - 1)
    max_series   = max(max_series, 1)
    series_len   = 0
    for i in range(series_start, min(series_start + max_series, n)):
        if out[i].isalpha():
            series_len += 1
        else:
            break
    if series_len == 0:
        out[series_start] = dict_int_to_char.get(out[series_start], out[series_start])
        series_len = 1

    # Unique number: remainder -> digit
    for i in range(series_start + series_len, n):
        out[i] = dict_char_to_int.get(out[i], out[i])

    return "".join(out)


def _fmt_standard(text: str) -> str:
    """
    Apply position-aware correction for the Standard Indian plate format:
      [LL][DD|D][LLL|LL|L][DDDD|DDD|DD|D]
    where L = letter slot, D = digit slot.

    Strategy:
      1. Pre-correct using _precorrect_standard (handles noisy OCR input).
      2. If the pre-corrected text matches the strict regex, split into the
         four logical segments and apply fine-grained correction.
      3. Otherwise fall back to the heuristic.
    """
    # First attempt: direct regex match (text already valid)
    m = re.fullmatch(
        r"([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})",
        text,
    )
    if m:
        state  = _correct_segment(m.group(1), "letter")
        rto    = _correct_segment(m.group(2), "digit")
        series = _correct_segment(m.group(3), "letter")
        num    = _correct_segment(m.group(4), "digit")
        return state + rto + series + num

    # Second attempt: pre-correct then re-match
    pre = _precorrect_standard(text)
    m2  = re.fullmatch(
        r"([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})",
        pre,
    )
    if m2:
        state  = _correct_segment(m2.group(1), "letter")
        rto    = _correct_segment(m2.group(2), "digit")
        series = _correct_segment(m2.group(3), "letter")
        num    = _correct_segment(m2.group(4), "digit")
        return state + rto + series + num

    return pre  # best effort


def _precorrect_bh(text: str) -> str:
    """
    Pre-correct a noisy string that is *approximately* a BH-series plate.
      pos 0-1   → digit  (year)
      pos 2-3   → letter (BH)
      pos 4-7   → digit  (4-digit number)
      pos 8+    → letter (1-2 letter suffix)
    """
    if len(text) < 6:
        return text
    out = list(text)
    for i in range(min(2, len(out))):
        out[i] = dict_char_to_int.get(out[i], out[i])
    # positions 2-3 should be 'B' and 'H' — keep as letters
    for i in range(2, min(4, len(out))):
        out[i] = dict_int_to_char.get(out[i], out[i])
    for i in range(4, min(8, len(out))):
        out[i] = dict_char_to_int.get(out[i], out[i])
    for i in range(8, len(out)):
        out[i] = dict_int_to_char.get(out[i], out[i])
    return "".join(out)


def _fmt_bh(text: str) -> str:
    """
    Apply position-aware correction for the BH-series plate format:
      [DD][BH][DDDD][LL|L]
    """
    # Direct match
    m = re.fullmatch(r"(\d{2})(BH)(\d{4})([A-Z]{1,2})", text)
    if m:
        yr     = _correct_segment(m.group(1), "digit")
        bh     = "BH"
        num    = _correct_segment(m.group(3), "digit")
        suffix = _correct_segment(m.group(4), "letter")
        return yr + bh + num + suffix

    # Pre-correct then retry
    pre = _precorrect_bh(text)
    m2  = re.fullmatch(r"(\d{2})(BH)(\d{4})([A-Z]{1,2})", pre)
    if m2:
        yr     = _correct_segment(m2.group(1), "digit")
        bh     = "BH"
        num    = _correct_segment(m2.group(3), "digit")
        suffix = _correct_segment(m2.group(4), "letter")
        return yr + bh + num + suffix

    return pre  # best effort


def _correct_segment(segment: str, expected: str) -> str:
    """
    Correct every character in *segment* according to *expected* type.

    Parameters
    ----------
    segment : str   – the substring to correct
    expected : str  – "letter" or "digit"
    """
    out = []
    for ch in segment:
        if expected == "letter":
            out.append(dict_int_to_char.get(ch, ch))
        else:  # digit
            out.append(dict_char_to_int.get(ch, ch))
    return "".join(out)


def _apply_correction_heuristic(text: str) -> str:
    """
    Fallback: iterate over *text* and correct each character based on whether
    a digit or letter is expected at that position according to a simple
    left-to-right scan of the Standard format.  Used when regex grouping fails
    (e.g. partially malformed plate).
    """
    # Positions 0-1 → letter, 2-3 → digit, then alternating letter/digit
    # This is intentionally a rough heuristic for edge cases.
    corrected = list(text)
    for i, ch in enumerate(corrected):
        if i < 2:
            corrected[i] = dict_int_to_char.get(ch, ch)
        elif i < 4:
            corrected[i] = dict_char_to_int.get(ch, ch)
        elif ch.isdigit():
            corrected[i] = dict_int_to_char.get(ch, ch)
        else:
            corrected[i] = dict_char_to_int.get(ch, ch)
    return "".join(corrected)


# Compiled format table — order matters: more specific patterns first.
# To add a new variant: append a dict with "name", "regex", and "formatter".
# BH-series is listed first because its pattern is a strict subset that would
# otherwise partially match the Standard pattern on some inputs.
_PLATE_FORMATS: list[dict] = [
    {
        "name": "BH-series",
        # 2 digits + BH + 4 digits + 1-2 uppercase letters
        "regex": re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$"),
        "fuzzy_precorrect": _precorrect_bh,
        "formatter": _fmt_bh,
    },
    {
        "name": "Standard Indian",
        # 2 letters (state) + 1-2 digits (RTO) + 1-3 letters (series) + 1-4 digits
        "regex": re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$"),
        "fuzzy_precorrect": _precorrect_standard,
        "formatter": _fmt_standard,
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def license_complies_format(text: str) -> bool:
    """
    Return True if *text* (already uppercased, spaces stripped) matches any
    registered Indian registration format — either directly, or after a
    pre-correction pass that fixes common OCR character confusions.

    Parameters
    ----------
    text : str – cleaned OCR output

    Returns
    -------
    bool

    Examples
    --------
    >>> license_complies_format("MH12AB1234")
    True
    >>> license_complies_format("22BH1234AA")
    True
    >>> license_complies_format("INVALID")
    False
    """
    for fmt in _PLATE_FORMATS:
        if fmt["regex"].fullmatch(text):
            return True
        # Also accept strings that become valid after pre-correction
        pre = fmt.get("fuzzy_precorrect", lambda x: x)(text)
        if pre != text and fmt["regex"].fullmatch(pre):
            return True
    return False


def format_license(text: str) -> str:
    """
    Apply position-aware character correction to *text* using the matched
    registration format's formatter.  If no format matches directly, a
    pre-correction pass is attempted first.

    Parameters
    ----------
    text : str – uppercased, spaces-stripped OCR text

    Returns
    -------
    str – corrected plate string
    """
    for fmt in _PLATE_FORMATS:
        if fmt["regex"].fullmatch(text):
            return fmt["formatter"](text)
        # Try pre-correction first, then format
        pre = fmt.get("fuzzy_precorrect", lambda x: x)(text)
        if fmt["regex"].fullmatch(pre):
            return fmt["formatter"](pre)
    return text


def read_license_plate(license_plate_crop: np.ndarray) -> tuple[Optional[str], Optional[float]]:
    """
    Run EasyOCR on *license_plate_crop*, normalize the output, validate the
    format, and return the best match.

    Processing steps per detection
    --------------------------------
    1. Convert text to uppercase.
    2. Strip spaces and non-alphanumeric characters (noise from plate borders,
       hyphens, dots that OCR sometimes hallucinates).
    3. Check format compliance.
    4. If valid, apply position-aware character correction and return.

    Parameters
    ----------
    license_plate_crop : np.ndarray – BGR or grayscale crop of the plate

    Returns
    -------
    (text, score) if a valid plate is found, else (None, None).
    """
    detections = reader.readtext(license_plate_crop)

    for _, text, score in detections:
        # Normalize
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)  # remove spaces, hyphens, dots, etc.

        if license_complies_format(text):
            return format_license(text), float(score)

    return None, None


def get_car(
    license_plate: tuple,
    vehicle_track_ids: np.ndarray,
) -> tuple[float, float, float, float, int]:
    """
    Find which tracked vehicle the detected *license_plate* belongs to.

    A vehicle is considered a match if the plate bounding box lies *fully
    inside* the vehicle bounding box.  We track vehicles rather than plates
    because:
      - Vehicle detections (COCO model) are far more stable and consistent.
      - Plates are small, low-res, and sometimes partially occluded.
      - Anchoring to a stable car_id lets us accumulate OCR results over many
        frames and pick the best one (highest confidence).

    Parameters
    ----------
    license_plate : tuple – (x1, y1, x2, y2, score, class_id)
    vehicle_track_ids : np.ndarray – shape (N, 5) rows of [x1, y1, x2, y2, car_id]

    Returns
    -------
    (xcar1, ycar1, xcar2, ycar2, car_id) on match, else (-1, -1, -1, -1, -1).
    """
    x1, y1, x2, y2, _score, _cls = license_plate

    for row in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = row

        # Containment check: plate box must be strictly inside vehicle box.
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return float(xcar1), float(ycar1), float(xcar2), float(ycar2), int(car_id)

    return -1, -1, -1, -1, -1


def write_csv(results: dict, output_path: str) -> None:
    """
    Write the pipeline results to a CSV file.

    Only rows that contain 'car', 'license_plate', and 'text' sub-keys are
    written (i.e., frames where OCR succeeded and format validation passed).

    CSV columns
    -----------
    frame_nmr, car_id, car_bbox, license_plate_bbox,
    license_plate_bbox_score, license_number, license_number_score

    Parameters
    ----------
    results     : dict  – nested dict[frame_nmr][car_id] produced by main.py
    output_path : str   – path to the output CSV file
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_nmr",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            ],
        )
        writer.writeheader()

        for frame_nmr, frame_data in results.items():
            for car_id, car_data in frame_data.items():
                # Only write rows that have all required sub-keys
                if (
                    "car" in car_data
                    and "license_plate" in car_data
                    and "text" in car_data["license_plate"]
                ):
                    lp = car_data["license_plate"]
                    writer.writerow(
                        {
                            "frame_nmr": frame_nmr,
                            "car_id": car_id,
                            "car_bbox": "[{} {} {} {}]".format(
                                *[int(v) for v in car_data["car"]["bbox"]]
                            ),
                            "license_plate_bbox": "[{} {} {} {}]".format(
                                *[int(v) for v in lp["bbox"]]
                            ),
                            "license_plate_bbox_score": round(lp["bbox_score"], 4),
                            "license_number": lp["text"],
                            "license_number_score": round(lp["text_score"], 4),
                        }
                    )
