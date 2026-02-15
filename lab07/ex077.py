from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time
import sys
from typing import List

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger

try:
    from ex072 import interface
    from ex076 import create_panorama
except ImportError:
    from lab07.ex072 import interface
    from lab07.ex076 import create_panorama


def show_free_memory(stop: threading.Event, interval_sec: float = 0.5) -> None:
    while not stop.is_set():
        mem = psutil.virtual_memory()
        print(f"Free memory: {mem.available / (1024 ** 3):.2f} GB")
        time.sleep(interval_sec)


def _load_images(image_paths: List[str]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Could not load image: {path}")
        images.append(img)
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 7.7 panorama creation with optional live capture.")
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Capture frames via ex072 interface instead of loading --images.",
    )
    parser.add_argument(
        "--images",
        default="imgNR1.png,imgNR2.png",
        help="Comma-separated image paths for offline panorama stitching.",
    )
    parser.add_argument("--features", type=int, default=400)
    parser.add_argument("--nndr", type=float, default=0.75)
    parser.add_argument("--output", default="panorama.png")
    parser.add_argument("--metrics-csv", default="metrics/lab07_ex077_metrics.csv")
    parser.add_argument("--debug-memory", action="store_true", help="Print free memory while running.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    stop_event = threading.Event()
    thread_id = None
    if args.debug_memory:
        thread_id = threading.Thread(target=show_free_memory, args=(stop_event,))
        thread_id.start()
        print("DEBUG MEMORY MODE ON")

    try:
        if args.capture:
            imgs = interface()
            if not imgs:
                raise RuntimeError("Capture returned no frames.")
        else:
            paths = [p.strip() for p in args.images.split(",") if p.strip()]
            imgs = _load_images(paths)

        if len(imgs) < 2:
            raise RuntimeError("Need at least two frames/images to build a panorama.")

        metrics = RuntimeMetricsLogger(args.metrics_csv, source="lab07_ex077", target_fps=None)

        started = metrics.begin_frame()
        _, _, _, result = create_panorama(imgs[0], imgs[1], features=args.features, nndr_ratio=args.nndr)
        metrics.log_frame(0, started, note="stitch_pair_1")

        for idx, next_img in enumerate(imgs[2:], start=1):
            started = metrics.begin_frame()
            _, _, _, result = create_panorama(result, next_img, features=args.features, nndr_ratio=args.nndr)
            metrics.log_frame(idx, started, note=f"stitch_pair_{idx+1}")

        cv2.imwrite(args.output, result)
        print(f"Panorama saved to: {args.output}")
        print(f"Metrics saved to: {args.metrics_csv}")
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    finally:
        if args.debug_memory and thread_id is not None:
            stop_event.set()
            thread_id.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
