from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger, open_video_source, parse_resolution

INFO = {
    "start": "Starting panorama capture program.",
    "init": "Initializing video source...",
    "init_ok": "Initialization succeeded.",
    "init_fail": "A critical error occurred.",
    "settings_ok": "Settings confirmed. Starting capture session.",
    "exit_ok": "Capture session closed.",
}


def capture_sequence(
    resolution: Tuple[int, int],
    frame_count: int,
    source: str = "auto",
    camera_index: int = 0,
    video_path: str = "",
    show_window: bool = True,
    metrics_csv: str = "metrics/lab07_ex072_metrics.csv",
) -> Optional[List[np.ndarray]]:
    print(INFO["init"])

    prefer_camera = source in {"camera", "auto"}
    fallback_video = video_path if source in {"auto", "camera", "video"} else None

    try:
        cap, source_label, msg = open_video_source(
            camera_index=camera_index,
            fallback_video=fallback_video,
            resolution=resolution,
            prefer_camera=prefer_camera,
        )
        print(msg)
        print(INFO["init_ok"])
    except Exception as exc:
        print(f"{INFO['init_fail']} {exc}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0
    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab07_ex072:{source_label}", target_fps=fps)

    frames: List[np.ndarray] = []
    frame_index = 0

    try:
        while len(frames) < frame_count:
            started = metrics.begin_frame()
            ret, frame = cap.read()
            if not ret:
                metrics.log_frame(frame_index, started, note="stream_end", dropped_frames=1)
                break

            if show_window:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    f"Captured {len(frames)}/{frame_count} | SPACE=capture | q=quit",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("panorama", preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    metrics.log_frame(frame_index, started, note="user_exit")
                    return frames
                if key == ord(" "):
                    frames.append(frame[:, :, :3].copy())
                    print(f"Picture taken! ({frame_count - len(frames)} remaining.)")
                    metrics.log_frame(frame_index, started, note="captured")
                else:
                    metrics.log_frame(frame_index, started, note="preview")
            else:
                # Headless mode: capture at 1 frame per second.
                frames.append(frame[:, :, :3].copy())
                metrics.log_frame(frame_index, started, note="captured_headless")
                time.sleep(1.0)

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(INFO["exit_ok"])

    return frames


def interface(
    resolution: Tuple[int, int] | None = None,
    frame_number: int | None = None,
    source: str = "auto",
    camera_index: int = 0,
    video_path: str = "",
    show_window: bool = True,
    metrics_csv: str = "metrics/lab07_ex072_metrics.csv",
):
    print(INFO["start"])
    if resolution is None:
        resolution = (640, 480)
    if frame_number is None:
        frame_number = 4

    print(INFO["settings_ok"])
    captured_frames = capture_sequence(
        resolution=resolution,
        frame_count=frame_number,
        source=source,
        camera_index=camera_index,
        video_path=video_path,
        show_window=show_window,
        metrics_csv=metrics_csv,
    )

    if captured_frames is None:
        return None

    if captured_frames:
        print(f"Successfully captured {len(captured_frames)} frames.")
    else:
        print("Capture cancelled by user.")

    return captured_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture panorama frames with fallback and metrics.")
    parser.add_argument("--resolution", default="640,480", help="Capture resolution as W,H.")
    parser.add_argument("--frame-count", type=int, default=4, help="Number of frames to capture.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="", help="Fallback or explicit video path.")
    parser.add_argument("--metrics-csv", default="metrics/lab07_ex072_metrics.csv")
    parser.add_argument("--no-window", action="store_true", help="Headless auto-capture mode.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        resolution = parse_resolution(args.resolution)
    except Exception as exc:
        print(f"Error: invalid resolution - {exc}")
        return 1

    frames = interface(
        resolution=resolution,
        frame_number=max(1, args.frame_count),
        source=args.source,
        camera_index=args.camera_index,
        video_path=args.video,
        show_window=not args.no_window,
        metrics_csv=args.metrics_csv,
    )
    return 0 if frames is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
