# Embedded Vision Prototyping And Native Optimization Workflow

This repository started as a sequence of computer vision labs (`lab03` to `lab09`) and is now being shaped into an embedded-vision engineering project that shows a practical Python-to-C optimization workflow.

The core story is no longer just "run OpenCV on a Raspberry Pi." It is:

- prototype image-processing ideas quickly in Python
- identify stable kernels worth optimizing
- re-implement the hot path in portable C
- validate native behavior on the host before hardware deployment
- keep the design ready for later Raspberry Pi reintegration

## Repository Structure

- `lab03`: Pillow fundamentals and chessboard image generation
- `lab04`: Camera/image preprocessing, grayscale conversion, and histograms
- `lab05`: Gradient, Sobel, Laplace, and LoG edge analysis
- `lab06`: Harris corner detection and response visualization
- `lab07`: Panorama pipeline (capture, SIFT, matching, stitching)
- `lab08`: Haar cascade object detection exercises
- `lab09`: Background subtraction and background replacement pipeline
- `csrc`: portable C kernels for host-side embedded practice
- `benchmarks`: host-side correctness and timing comparisons for Python vs native C
- `docs/banner.png`: project banner image
- `docs/embedded-roadmap.md`: embedded-systems extension plan for this repo
- `docs/python-to-c-workflow.md`: portfolio-facing framing for the optimization story

## Engineering Narrative

If someone reads this repo as a portfolio project, the intended takeaway is:

- Python is used as the rapid-prototyping layer for image-processing experiments.
- C is used where the algorithm is stable enough to justify lower-level optimization work.
- Host-side build and test steps are used before touching embedded hardware.
- Raspberry Pi becomes the deployment target, not the only place where development can happen.

That framing makes the project read more like engineering iteration and system design, and less like a one-off course exercise.

## Reproducible Python Setup

Use Python 3.9+ and a virtual environment.

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

## Native C Track

You can practice embedded-style native development on your laptop without Raspberry Pi hardware by building the small C module under `csrc/`.

```bash
make -C csrc test
python3 benchmarks/benchmark_native_vs_python.py
```

It currently covers convolution, Sobel, Laplace, a compact Harris response, and alpha blending that align with `lab05`, `lab06`, and `lab09`. The Python bridge lives in `embedded_utils/native_image_ops.py`, and the lab helpers can now call the native path directly for selected kernels.

## Python-To-C Workflow

This repo now supports a cleaner optimization narrative:

1. Develop and validate the algorithm in Python.
2. Extract the most reusable or performance-sensitive kernel.
3. Port that kernel to portable C with explicit memory layout and status handling.
4. Test the C implementation on the host.
5. Later integrate the native path into embedded deployment.

For project-positioning notes, see `docs/python-to-c-workflow.md`.

## Measured Performance

Host-side benchmark run on April 9, 2026 using `benchmarks/benchmark_native_vs_python.py` on an `arm64` macOS host (`Darwin 25.4.0`) with Python `3.9.23`.

These are not Raspberry Pi numbers, but they make the optimization claim concrete and reproducible while hardware is unavailable.

| Operation | Input | Correctness | Python median (ms) | Native C median (ms) | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| Lab 05 Sobel magnitude | 1024x1024 grayscale | max abs diff = 0.0001 | 32.16 | 8.16 | 3.94x |
| Lab 05 Laplace filter | 1024x1024 grayscale | max abs diff = 0.0000 | 7.60 | 1.54 | 4.93x |
| Lab 09 alpha blend | 320x320 over 720x1280 | max channel diff = 1 | 0.49 | 0.18 | 2.66x |

The small alpha-blend difference is a 1-count uint8 rounding gap on some pixels; the benchmark script reports that explicitly so the comparison stays honest.

Once Raspberry Pi hardware is back in the loop, the same benchmark flow can be rerun on-device and extended with a second table for deployment-target measurements.

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
