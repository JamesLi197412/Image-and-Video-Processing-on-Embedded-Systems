from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys
from typing import List

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger, open_video_source


def _try_load_cascade(path: str) -> cv2.CascadeClassifier | None:
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        return None
    return cascade


def _load_cascades(vertical_path: str, horizontal_path: str) -> List[cv2.CascadeClassifier]:
    cascades: List[cv2.CascadeClassifier] = []
    v = _try_load_cascade(vertical_path)
    if v is not None:
        cascades.append(v)
    else:
        print(f"Warning: vertical cascade missing/unreadable: {vertical_path}")

    h = _try_load_cascade(horizontal_path)
    if h is not None:
        cascades.append(h)
    else:
        print(f"Warning: horizontal cascade missing/unreadable: {horizontal_path}")

    if cascades:
        return cascades

    fallback = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    f = _try_load_cascade(fallback)
    if f is None:
        raise RuntimeError("No valid cascades available (including OpenCV fallback).")
    print(f"Warning: using fallback cascade: {fallback}")
    return [f]


def run_detection(
    source: str,
    camera_index: int,
    video_path: str,
    vertical_cascade: str,
    horizontal_cascade: str,
    metrics_csv: str,
    scale_factor: float,
    min_neighbors: int,
    show_window: bool,
) -> int:
    prefer_camera = source in {"camera", "auto"}
    fallback_video = video_path if source in {"auto", "video", "camera"} else None

    cap, source_label, msg = open_video_source(
        camera_index=camera_index,
        fallback_video=fallback_video,
        resolution=None,
        prefer_camera=prefer_camera,
    )
    print(msg)

    cascades = _load_cascades(vertical_cascade, horizontal_cascade)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0
    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab08_ex083:{source_label}", target_fps=fps)

    frame_idx = 0
    while True:
        start = metrics.begin_frame()
        ret, frame = cap.read()
        if not ret:
            metrics.log_frame(frame_idx, start, note="stream_end", dropped_frames=1)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        for cascade in cascades:
            objs = cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            detections.append(objs)

        count = 0
        for x, y, w, h in itertools.chain.from_iterable(detections):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

        if show_window:
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                metrics.log_frame(frame_idx, start, note="user_exit")
                break

        metrics.log_frame(frame_idx, start, note=f"detections={count}")
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 8.3 object detection with cascade fallback and metrics.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="my_video.avi")
    parser.add_argument("--vertical-cascade", default="pen_vertical.xml")
    parser.add_argument("--horizontal-cascade", default="pen_horizontal.xml")
    parser.add_argument("--scale-factor", type=float, default=1.7)
    parser.add_argument("--min-neighbors", type=int, default=25)
    parser.add_argument("--metrics-csv", default="metrics/lab08_ex083_metrics.csv")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_detection(
            source=args.source,
            camera_index=args.camera_index,
            video_path=args.video,
            vertical_cascade=args.vertical_cascade,
            horizontal_cascade=args.horizontal_cascade,
            metrics_csv=args.metrics_csv,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            show_window=not args.no_window,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
