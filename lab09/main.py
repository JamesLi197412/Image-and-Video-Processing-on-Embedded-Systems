from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from .grabcut import grab_cut
    from .alphachannel import add_alpha_channel
    from .pasteImage import paste_image
except ImportError:
    from grabcut import grab_cut
    from alphachannel import add_alpha_channel
    from pasteImage import paste_image

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger, open_video_source


def _load_cascade(cascade_path: str) -> tuple[cv2.CascadeClassifier | None, str]:
    direct = cv2.CascadeClassifier(cascade_path)
    if not direct.empty():
        return direct, f"Loaded cascade: {cascade_path}"

    fallback_path = str(Path(cv2.data.haarcascades) / Path(cascade_path).name)
    fallback = cv2.CascadeClassifier(fallback_path)
    if not fallback.empty():
        return fallback, f"Loaded fallback OpenCV cascade: {fallback_path}"

    return None, (
        "Cascade unavailable; falling back to central ROI compositing. "
        "Provide a valid cascade with --cascade for better quality."
    )


def _fallback_rect(width: int, height: int) -> tuple[int, int, int, int]:
    w = int(width * 0.5)
    h = int(height * 0.7)
    x = max(0, (width - w) // 2)
    y = max(0, (height - h) // 2)
    return x, y, w, h


def background_replacement(
    source_mode: str,
    foreground_video_path: str,
    background_video_path: str,
    cascade_path: str,
    output_path: str,
    metrics_csv: str,
    camera_index: int,
    scale_factor: float,
    min_neighbors: int,
    min_size: tuple[int, int],
) -> None:
    # Foreground source: camera preferred or file.
    if source_mode == "camera":
        cap, fg_source, source_msg = open_video_source(
            camera_index=camera_index,
            fallback_video=foreground_video_path,
            resolution=None,
            prefer_camera=True,
        )
        print(source_msg)
    else:
        cap = cv2.VideoCapture(foreground_video_path)
        fg_source = "video"
        if not cap.isOpened():
            raise RuntimeError(f"Foreground video does not exist/cannot be opened: {foreground_video_path}")

    bg_cap = cv2.VideoCapture(background_video_path)
    if not bg_cap.isOpened():
        raise RuntimeError(f"Background video does not exist/cannot be opened: {background_video_path}")

    cascade, cascade_msg = _load_cascade(cascade_path)
    print(cascade_msg)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab09_main:{fg_source}", target_fps=fps)

    frame_index = 0
    while True:
        frame_start = metrics.begin_frame()
        ret, frame = cap.read()
        success, background_frame = bg_cap.read()

        if not ret:
            metrics.log_frame(frame_index, frame_start, note="foreground_end", dropped_frames=1)
            break

        if not success:
            # Loop background video for long foreground streams.
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, background_frame = bg_cap.read()
            if not success:
                metrics.log_frame(frame_index, frame_start, note="background_read_fail", dropped_frames=1)
                break

        if background_frame.shape[:2] != (height, width):
            background_frame = cv2.resize(background_frame, (width, height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if cascade is not None:
            detected_objects = cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
            )
        else:
            detected_objects = []

        if len(detected_objects) == 0:
            detected_objects = [_fallback_rect(width, height)]
            note = "fallback_rect"
        else:
            note = "cascade_detected"

        x, y, w, h = detected_objects[0]
        gc = grab_cut(frame, x, y, w, h, iter_count=5)
        cropped = gc[y : y + h, x : x + w]
        ag = add_alpha_channel(cropped)
        background_frame = paste_image(ag, background_frame, x, y, w, h)

        out.write(background_frame)
        metrics.log_frame(frame_index, frame_start, note=note)
        frame_index += 1

    cap.release()
    bg_cap.release()
    out.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab09 background replacement with fallback and runtime metrics.")
    parser.add_argument("--source", choices=["video", "camera"], default="video")
    parser.add_argument("--foreground-video", default="fg.mp4", help="Foreground video path or camera fallback.")
    parser.add_argument("--background-video", default="bg.mp4", help="Background replacement video path.")
    parser.add_argument("--cascade", default="haarcascade_frontalface_default.xml", help="Cascade file path.")
    parser.add_argument("--output", default="out.mp4", help="Output video path.")
    parser.add_argument("--metrics-csv", default="metrics/lab09_main_metrics.csv", help="CSV runtime metrics path.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index if --source camera.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Cascade scaleFactor.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Cascade minNeighbors.")
    parser.add_argument("--min-size", default="30,30", help="Cascade min size as W,H.")
    return parser.parse_args()


def _parse_min_size(text: str) -> tuple[int, int]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError("--min-size must be W,H")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError("--min-size values must be positive")
    return w, h


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    try:
        min_size = _parse_min_size(args.min_size)
        background_replacement(
            source_mode=args.source,
            foreground_video_path=args.foreground_video,
            background_video_path=args.background_video,
            cascade_path=args.cascade,
            output_path=args.output,
            metrics_csv=args.metrics_csv,
            camera_index=args.camera_index,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=min_size,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.3f} s")
    print(f"Output saved to: {args.output}")
    print(f"Metrics saved to: {args.metrics_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
