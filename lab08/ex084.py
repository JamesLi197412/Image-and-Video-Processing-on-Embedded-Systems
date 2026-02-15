from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger
from embedded_utils import open_video_source

try:
    from ex083 import _load_cascades
except ImportError:
    from lab08.ex083 import _load_cascades


def run_detection_and_export(
    source: str,
    camera_index: int,
    video_path: str,
    vertical_cascade: str,
    horizontal_cascade: str,
    output_dir: str,
    metrics_csv: str,
    scale_factor: float,
    min_neighbors: int,
    show_window: bool,
) -> int:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab08_ex084:{source_label}", target_fps=fps)

    frame_id = 0
    while True:
        start = metrics.begin_frame()
        ret, frame = cap.read()
        if not ret:
            metrics.log_frame(frame_id, start, note="stream_end", dropped_frames=1)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        total_detections = 0
        for cascade in cascades:
            detections = cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            for x, y, w, h in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                total_detections += 1

        if total_detections > 0:
            cv2.imwrite(str(out_dir / f"frame_{frame_id:05d}.png"), frame)

        if show_window:
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                metrics.log_frame(frame_id, start, note="user_exit")
                break

        metrics.log_frame(frame_id, start, note=f"detections={total_detections}")
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection frames saved to: {out_dir.resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 8.4 detection + frame export with fallback and metrics.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="video")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="my_video.avi")
    parser.add_argument("--vertical-cascade", default="pen_vertical.xml")
    parser.add_argument("--horizontal-cascade", default="pen_horizontal.xml")
    parser.add_argument("--scale-factor", type=float, default=1.7)
    parser.add_argument("--min-neighbors", type=int, default=25)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--metrics-csv", default="metrics/lab08_ex084_metrics.csv")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_detection_and_export(
            source=args.source,
            camera_index=args.camera_index,
            video_path=args.video,
            vertical_cascade=args.vertical_cascade,
            horizontal_cascade=args.horizontal_cascade,
            output_dir=args.output_dir,
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
