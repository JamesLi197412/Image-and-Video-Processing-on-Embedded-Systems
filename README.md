# Embedded Image and Video Processing Labs (Raspberry Pi)

This repository contains a sequence of computer vision labs (`lab03` to `lab09`) focused on classic image processing and embedded vision workflows on Raspberry Pi.

## Repository Structure

- `lab03`: Pillow fundamentals and chessboard image generation
- `lab04`: Camera/image preprocessing, grayscale conversion, and histograms
- `lab05`: Gradient, Sobel, Laplace, and LoG edge analysis
- `lab06`: Harris corner detection and response visualization
- `lab07`: Panorama pipeline (capture, SIFT, matching, stitching)
- `lab08`: Haar cascade object detection exercises
- `lab09`: Background subtraction and background replacement pipeline
- `docs/banner.png`: project banner image

## Reproducible Python Setup

Use Python 3.10+ and a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Raspberry Pi camera/button labs (`lab07`, `lab08`, `lab09`):

```bash
pip install -r requirements-rpi.txt
```

For tests/dev tools:

```bash
pip install -r requirements-dev.txt
```

## Preflight Validation

Run preflight checks before field runs:

```bash
python3 preflight_check.py \
  --require-camera \
  --cascade lab09/haarcascade_frontalface_default.xml \
  --video lab09/fg.mp4 \
  --video lab09/bg.mp4
```

## Raspberry Pi Board Setup Checklist

1. Use Raspberry Pi OS (Bookworm recommended) and update packages.

```bash
sudo apt update && sudo apt upgrade -y
```

2. Install camera and GPIO system packages.

```bash
sudo apt install -y python3-opencv python3-picamera2 python3-libcamera python3-gpiozero
```

3. Add your user to required groups, then re-login.

```bash
sudo usermod -aG video,gpio $USER
```

4. Validate camera stack.

```bash
libcamera-hello -t 2000
```

5. Confirm enough power and cooling for long runs.
- Use a stable 5V/3A PSU (or better for your board model).
- Use a heatsink/fan to avoid thermal throttling.

6. Keep lab data organized.
- Run each script from its own lab folder.
- Keep generated outputs in per-lab subfolders (for example `output/`).

## How to Run

Most scripts are standalone and use relative paths. Run them from their own lab directory.

```bash
cd lab04
python3 ex42.py
```

## Lab Index

- [Lab 03 README](lab03/README.md)
- [Lab 04 README](lab04/README.md)
- [Lab 05 README](lab05/README.md)
- [Lab 06 README](lab06/README.md)
- [Lab 07 README](lab07/README.md)
- [Lab 08 README](lab08/README.md)
- [Lab 09 README](lab09/README.md)

## Notes

- Several later labs expect Raspberry Pi hardware (camera + GPIO buttons).
- Some scripts expect local input assets (for example `imgNR1.png`, `imgNR2.png`, `fg.mp4`, `bg.mp4`, cascade XML files).
- Output images/videos are written into the same lab folder unless otherwise noted in the script.
- Runtime metrics CSV files are written under `metrics/` for upgraded realtime scripts.
- Deployment/systemd details: `docs/deployment.md` and `docs/systemd/embedded-vision.service`.
