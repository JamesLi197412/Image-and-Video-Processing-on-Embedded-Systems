# Raspberry Pi Deployment Guide

## 1. Hardware Wiring and GPIO Map

Current lab scripts use the following GPIO map:

- `GPIO 5`: capture trigger button (legacy panorama flows)
- `GPIO 6`: stop/exit button (legacy camera flows)

Recommended wiring:

- One side of each momentary push button to GPIO pin.
- Other side to GND.
- Use internal pull-up/down in software where needed.

## 2. Power Requirements

- Raspberry Pi 4: stable `5V / 3A` power supply minimum.
- Raspberry Pi 5: stable `5V / 5A` official PSU recommended for camera + sustained CPU load.
- Avoid powering camera + USB peripherals from weak hubs during long runs.

## 3. Thermal Limits and Cooling

- Continuous CV workloads can throttle CPU/GPU when board temperature approaches ~80C.
- Use at least a heatsink; fan recommended for multi-minute pipelines.
- Monitor thermal state:

```bash
vcgencmd measure_temp
vcgencmd get_throttled
```

## 4. Software Bring-Up Checklist

```bash
cd /home/pi/pythonlab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-rpi.txt
```

Install system camera stack on Raspberry Pi OS:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera python3-gpiozero
```

Run preflight checks:

```bash
python3 preflight_check.py \
  --require-camera \
  --cascade lab09/haarcascade_frontalface_default.xml \
  --video lab09/fg.mp4 \
  --video lab09/bg.mp4
```

## 5. Expected FPS (Rule-of-Thumb)

These are rough expectations at `640x480` with basic OpenCV ops, assuming adequate cooling:

- Raspberry Pi 4 (4GB/8GB): ~8-20 FPS depending on algorithm and cascade load.
- Raspberry Pi 5: ~15-35 FPS for similar pipelines.

Actual FPS depends on:

- model + clocks,
- temperature/throttling,
- camera driver path,
- algorithm settings (`scaleFactor`, `minNeighbors`, GrabCut iterations),
- storage I/O if writing video and metrics.

## 6. systemd Autostart

Service template is provided at:

- `docs/systemd/embedded-vision.service`

Install steps:

```bash
sudo cp docs/systemd/embedded-vision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable embedded-vision.service
sudo systemctl start embedded-vision.service
sudo systemctl status embedded-vision.service
```

## 7. Field Debug Tips

- If camera fails, run `libcamera-hello -t 2000` first.
- If GPIO access fails, confirm user groups:

```bash
groups
```

Expected to include `video` and `gpio`.
- If FPS drops, inspect `metrics/*.csv` for latency spikes and dropped-frame estimates.
