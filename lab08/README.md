# Lab 08 - Object Detection with Haar Cascades

## Goal

Experiment with Haar-cascade detection on video streams, evaluate frame-rate tradeoffs, and combine detections with post-processing.

## Files

- `ex080.py`: notes/commands for training a cascade classifier
- `ex081.py`: grayscale video playback with quit controls
- `ex082.py`: capture loop with FPS estimation
- `ex083.py`: dual-cascade object detection (vertical/horizontal pen)
- `ex084.py`: detection + conditional frame export
- `ex085.py`: detection + Hough line orientation estimation

## Hardware/Runtime Requirements

- Raspberry Pi camera for live mode (`Picamera2`)
- GPIO red button on pin `6`
- Python packages: `opencv-python`, `numpy`, `picamera2`, `gpiozero`

## Run

```bash
cd lab08
python3 ex081.py --source auto --video my_video.avi
python3 ex082.py --source auto --resolution 640,480 --sample-frames 200
python3 ex083.py --source auto --video my_video.avi
python3 ex084.py --source auto --video my_video.avi --output-dir output
python3 ex085.py --source auto --video my_video.avi
```

## Expected Inputs

- Optional recorded video: `my_video.avi`
- Cascade files for pen detection (not included in this repo):
  - `pen_vertical.xml` / `pen_horizontal.xml` (for `ex083.py`, `ex084.py`)
  - `pen_vertical_classifier.xml` (for `ex085.py`)

## Outputs

- Real-time display windows
- `output/frame_*.png` from `ex084.py` when detections occur

## Notes

- This README only covers Lab 08. Lab 09 has its own README in `lab09/README.md`.
- If custom pen cascades are missing, scripts degrade to OpenCV fallback cascade and print warnings.
- Realtime scripts write CSV metrics in `metrics/`.
