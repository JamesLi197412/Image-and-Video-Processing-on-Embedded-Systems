from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import open_video_source, parse_resolution


def take_img(resolution: tuple[int, int], source: str, camera_index: int, video_path: str) -> tuple[cv2.typing.MatLike, str]:
    prefer_camera = source in {"camera", "auto"}
    fallback_video = video_path if source in {"auto", "camera", "video"} else None

    cap, source_label, msg = open_video_source(
        camera_index=camera_index,
        fallback_video=fallback_video,
        resolution=resolution,
        prefer_camera=prefer_camera,
    )
    print(msg)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to capture frame from source.")
    return frame, source_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one test frame for panorama experiments.")
    parser.add_argument("--resolution", default="640,480", help="Resolution as W,H.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="", help="Fallback or explicit video path.")
    parser.add_argument("--output", default="", help="Output file path. Defaults to test_image_<w>_<h>.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        resolution = parse_resolution(args.resolution)
        frame, source_label = take_img(
            resolution=resolution,
            source=args.source,
            camera_index=args.camera_index,
            video_path=args.video,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    output = args.output or f"test_image_{resolution[0]}_{resolution[1]}.png"
    cv2.imwrite(output, frame)
    print(f"Saved frame from {source_label}: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
