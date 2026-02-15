from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger, open_video_source


def run_background_subtraction(
    source_mode: str,
    camera_index: int,
    video_path: str,
    algorithm: str,
    metrics_csv: str,
    show_window: bool,
) -> int:
    prefer_camera = source_mode in {"camera", "auto"}
    fallback_video = video_path if source_mode in {"auto", "video", "camera"} else None

    cap, source_label, source_msg = open_video_source(
        camera_index=camera_index,
        fallback_video=fallback_video,
        resolution=None,
        prefer_camera=prefer_camera,
    )
    print(source_msg)

    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400.0, detectShadows=True)
    subtractor = knn if algorithm == "knn" else mog2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0
    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab09_ex093:{source_label}", target_fps=fps)

    frame_index = 0
    while True:
        started = metrics.begin_frame()
        ret, frame = cap.read()
        if not ret:
            metrics.log_frame(frame_index, started, note="stream_end", dropped_frames=1)
            break

        fgmask = subtractor.apply(frame)

        if show_window:
            cv2.imshow("Frame", frame)
            cv2.imshow("FG Mask", fgmask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                metrics.log_frame(frame_index, started, note="user_exit")
                break

        metrics.log_frame(frame_index, started, note=f"algo={algorithm}")
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 9.3 background subtraction with fallback and metrics.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="", help="Fallback or explicit video source path.")
    parser.add_argument("--algorithm", choices=["mog2", "knn"], default="mog2")
    parser.add_argument("--metrics-csv", default="metrics/lab09_ex093_metrics.csv")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV display windows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_background_subtraction(
            source_mode=args.source,
            camera_index=args.camera_index,
            video_path=args.video,
            algorithm=args.algorithm,
            metrics_csv=args.metrics_csv,
            show_window=not args.no_window,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
