from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger, open_video_source


def run_face_detection(
    source_mode: str,
    camera_index: int,
    video_path: str,
    cascade_path: str,
    scale_factor: float,
    min_neighbors: int,
    min_size: tuple[int, int],
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

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        fallback_path = str(Path(cv2.data.haarcascades) / Path(cascade_path).name)
        face_cascade = cv2.CascadeClassifier(fallback_path)
        if face_cascade.empty():
            raise RuntimeError(
                f"Could not load cascade from '{cascade_path}' or fallback '{fallback_path}'."
            )
        print(f"Using fallback cascade: {fallback_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0

    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab09_ex091:{source_label}", target_fps=fps)

    frame_index = 0
    while True:
        started = metrics.begin_frame()
        ret, frame = cap.read()
        if not ret:
            metrics.log_frame(frame_index, started, note="stream_end", dropped_frames=1)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if show_window:
            cv2.imshow("Haar Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                metrics.log_frame(frame_index, started, note="user_exit")
                break

        metrics.log_frame(frame_index, started, note=f"faces={len(faces)}")
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


def _parse_min_size(text: str) -> tuple[int, int]:
    parts = text.replace(" ", "").split(",")
    if len(parts) != 2:
        raise ValueError("--min-size must be W,H")
    w, h = int(parts[0]), int(parts[1])
    if w <= 0 or h <= 0:
        raise ValueError("--min-size values must be positive")
    return w, h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 9.1 face detection with fallback and metrics.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="", help="Fallback or explicit video source path.")
    parser.add_argument("--cascade", default="haarcascade_frontalface_default.xml")
    parser.add_argument("--scale-factor", type=float, default=1.1)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-size", default="30,30")
    parser.add_argument("--metrics-csv", default="metrics/lab09_ex091_metrics.csv")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV display window.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        min_size = _parse_min_size(args.min_size)
        return run_face_detection(
            source_mode=args.source,
            camera_index=args.camera_index,
            video_path=args.video,
            cascade_path=args.cascade,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_size=min_size,
            metrics_csv=args.metrics_csv,
            show_window=not args.no_window,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
