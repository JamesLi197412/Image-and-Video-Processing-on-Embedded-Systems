from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embedded_utils import RuntimeMetricsLogger, open_video_source


def hough_transform(x: int, y: int, w: int, h: int, image: np.ndarray, t_low: int, t_high: int):
    height, width = image.shape[:2]

    x1 = max(0, x - 200)
    y1 = max(0, y - 40)
    x2 = min(width, x + w + 200)
    y2 = min(height, y + h + 40)

    if x2 <= x1 or y2 <= y1:
        return None

    cropped_image = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray, t_low, t_high, apertureSize=3)

    lines = cv2.HoughLines(edge_image, 1, np.pi / 180, 120)
    if lines is None:
        return None

    rho, theta = lines[0][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1l = int(x0 + 1000 * (-b))
    y1l = int(y0 + 1000 * (a))
    x2l = int(x0 - 1000 * (-b))
    y2l = int(y0 - 1000 * (a))

    cv2.line(cropped_image, (x1l, y1l), (x2l, y2l), (0, 0, 255), 2)

    if theta > np.pi / 2:
        angle = (theta - np.pi / 2) * 180 / np.pi
    else:
        angle = theta * 180 / np.pi

    cv2.putText(
        image,
        f"Angle: {angle:.2f}",
        (10, 25),
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (0, 0, 255),
        2,
    )
    return angle


def _load_cascade(path: str) -> cv2.CascadeClassifier:
    cascade = cv2.CascadeClassifier(path)
    if not cascade.empty():
        return cascade

    fallback = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(fallback)
    if cascade.empty():
        raise RuntimeError(f"Unable to load cascade '{path}' and OpenCV fallback '{fallback}'.")
    print(f"Warning: using fallback cascade: {fallback}")
    return cascade


def run_detection_with_hough(
    source: str,
    camera_index: int,
    video_path: str,
    cascade_path: str,
    t_low: int,
    t_high: int,
    scale_factor: float,
    min_neighbors: int,
    metrics_csv: str,
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

    cascade = _load_cascade(cascade_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0
    metrics = RuntimeMetricsLogger(metrics_csv, source=f"lab08_ex085:{source_label}", target_fps=fps)

    frame_idx = 0
    while True:
        started = metrics.begin_frame()
        ret, frame = cap.read()
        if not ret:
            metrics.log_frame(frame_idx, started, note="stream_end", dropped_frames=1)
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(25, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        angle_note = ""
        for x, y, w, h in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            angle = hough_transform(x, y, w, h, frame, t_low=t_low, t_high=t_high)
            if angle is not None:
                angle_note = f",angle={angle:.2f}"

        if show_window:
            cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                metrics.log_frame(frame_idx, started, note="user_exit")
                break

        metrics.log_frame(frame_idx, started, note=f"detections={len(detections)}{angle_note}")
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 8.5 detection + Hough line angle estimation.")
    parser.add_argument("--source", choices=["auto", "camera", "video"], default="auto")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video", default="my_video.avi")
    parser.add_argument("--cascade", default="pen_vertical_classifier.xml")
    parser.add_argument("--t-low", type=int, default=50)
    parser.add_argument("--t-high", type=int, default=150)
    parser.add_argument("--scale-factor", type=float, default=1.7)
    parser.add_argument("--min-neighbors", type=int, default=25)
    parser.add_argument("--metrics-csv", default="metrics/lab08_ex085_metrics.csv")
    parser.add_argument("--no-window", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_detection_with_hough(
            source=args.source,
            camera_index=args.camera_index,
            video_path=args.video,
            cascade_path=args.cascade,
            t_low=args.t_low,
            t_high=args.t_high,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            metrics_csv=args.metrics_csv,
            show_window=not args.no_window,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
