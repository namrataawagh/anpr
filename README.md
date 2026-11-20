# Indian ANPR — How to Use

## Project structure

```
anpr/
├── main.py                        # Video pipeline (the main script)
├── detect_image.py                # Single photo detection
├── add_missing_data.py            # Fills gaps between detections
├── visualize.py                   # Renders annotated output video
├── util.py                        # Core logic (don't run directly)
├── requirements.txt
├── models/
│   └── license_plate_detector.pt  # Plate detection model
├── yolov8n.pt                     # Vehicle detection model (auto-downloaded)
└── sort/
    └── sort.py                    # Vehicle tracker
```

---

## One-time setup

```bash
cd /home/namrata/Documents/anpr
source venv/bin/activate      # activate virtual environment (always do this first)
```

All packages are already installed in the venv.

---

## Usage Mode 1 — Video (the full pipeline)

Use this for dashcam footage, CCTV video, highway recordings.

### Step 1 — Detect plates, save to CSV
```bash
python main.py --video your_video.mp4
```
Processes every frame: detects vehicles → tracks them → reads plates → writes `test.csv`.

Options:
```bash
python main.py --video your_video.mp4 \
               --output results.csv \     # change output CSV name
               --max-frames 500           # process only first 500 frames
               --debug                   # print what OCR reads for every plate
```

### Step 2 — Fill in missing frames
```bash
python add_missing_data.py
```
Reads `test.csv`, linearly interpolates bounding boxes for frames where
the plate wasn't detected, writes `test_interpolated.csv`.

### Step 3 — Render annotated video
```bash
python visualize.py
```
Reads `test_interpolated.csv` + original video, draws boxes and plate text,
saves `out.mp4`.

```bash
xdg-open out.mp4    # watch the result
```

---

## Usage Mode 2 — Single Photo

Use this for a photo of a car or a close-up plate shot.

```bash
# Photo of a car on the road
python detect_image.py --image photo.jpg

# Close-up of just the plate (no visible car body)
python detect_image.py --image plate.jpg --no-vehicle

# Save to a specific path
python detect_image.py --image photo.jpg --output annotated.jpg
```

Output is saved automatically as `photo_anpr.jpg` next to the input.

---

## What the output looks like

**In the terminal:**
```
✅  Plate: MH12AB1234  (OCR score: 94%, detector conf: 34%)
❌  Region: OCR read 'ABC 1234'(0.85) → failed Indian format validation
```

**In the annotated image/video:**
- Yellow corner brackets → tracked vehicle
- Red rectangle → license plate region
- White banner above → plate crop image + plate text + confidence score

---

## Supported plate formats

| Format | Example | Pattern |
|--------|---------|---------|
| Standard | MH12AB1234 | 2 letters + 1-2 digits + 1-3 letters + 1-4 digits |
| 1-digit RTO | DL8CAF1234 | Same, single-digit RTO |
| BH-series | 22BH1234AA | 2 digits + BH + 4 digits + 1-2 letters |

---

## Troubleshooting

**No plates in CSV / empty CSV**
- Non-Indian plates (US, UK etc.) are intentionally rejected by the Indian format validator
- Run with `--debug` to see what OCR is reading
- Try a lower confidence threshold: add `--conf 0.10` to detect_image.py

**Video shows vehicles but no plate overlays**
- The plate model found regions but OCR couldn't read them (blurry, angle, lighting)
- Use better quality footage or higher resolution

**"SORT tracker not found"**
```bash
git clone https://github.com/abewley/sort.git sort
```

**"License plate model not found"**
The model should be at `models/license_plate_detector.pt` (6MB, already downloaded).
If missing, re-download:
```bash
python -c "
import requests
r = requests.get('https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt')
open('models/license_plate_detector.pt','wb').write(r.content)
"
```
