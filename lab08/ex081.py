from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger, open_video_source


def run_video_grayscale(source: str, camera_index: int, video_path: str, metrics_csv: str, show_window: bool) -> int:
    prefer_camera = source in {"camera", "auto"}
    fallback_video = video_path if source in {"auto", "video", "camera"} else None

    cap, source_label, msg = open_video_source(
        camera_index=camera_index,
        fallback_video=fallback_video,
        resolution=None,
        prefer_camera=prefer_camera,
    )
    print(msg)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0
    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab08_ex081:{source_label}", target_fps=fps)

    frame_idx = 0
    while True:
        start = metrics.begin_frame()
        ret, frame = cap.read()
        if not ret:
            metrics.log_frame(frame_idx, start, note="stream_end", dropped_frames=1)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if show_window:
            cv2.imshow("video", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                metrics.log_frame(frame_idx, start, note="user_exit")
                break

        metrics.log_frame(frame_idx, start)
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 8.1 grayscale display with fallback and metrics.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="video")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="my_video.avi")
    parser.add_argument("--metrics-csv", default="metrics/lab08_ex081_metrics.csv")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_video_grayscale(
            source=args.source,
            camera_index=args.camera_index,
            video_path=args.video,
            metrics_csv=args.metrics_csv,
            show_window=not args.no_window,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
