# Lab 09 - Background Subtraction and Background Replacement

## Goal

Build a foreground extraction and compositing pipeline using Haar detection, GrabCut segmentation, alpha masking, and background replacement.

## Files

- `ex091.py`: face detection from camera/Picamera2 stream
- `ex093.py`: background subtraction (MOG2/KNN)
- `grabcut.py`: GrabCut helper
- `alphachannel.py`: convert BGR crop to BGRA with transparent black background
- `pasteImage.py`: alpha compositing helper
- `main.py`: end-to-end video background replacement pipeline
- `haarcascade_frontalface_default.xml`: face cascade weights

## Run

### 1) Live face detection

```bash
cd lab09
python3 ex091.py --source auto --cascade haarcascade_frontalface_default.xml
```

### 2) Live background subtraction

```bash
cd lab09
python3 ex093.py --source auto --algorithm mog2
```

### 3) Offline background replacement

Prepare input videos in this folder:

- `fg.mp4` (foreground/person video)
- `bg.mp4` (replacement background video)

Then run:

```bash
cd lab09
python3 main.py --source video --foreground-video fg.mp4 --background-video bg.mp4
```

## Output

- `out.mp4` written in `lab09`

## Dependencies

```bash
pip install numpy opencv-python picamera2 gpiozero
```

## Notes

- `main.py` uses `haarcascade_frontalface_default.xml` to find the object region before GrabCut.
- Inputs are processed frame-by-frame; output length is capped by the shorter of foreground/background videos.
- If camera is unavailable and `--source auto/camera` is used, scripts can fall back to `--video` or `--foreground-video`.
- Runtime metrics are written to CSV (`metrics/`).
