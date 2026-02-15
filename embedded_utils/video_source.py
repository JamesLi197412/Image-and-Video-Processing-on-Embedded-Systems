from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2


def parse_resolution(text: str) -> Tuple[int, int]:
    value = text.strip().lower().replace("x", ",").replace(" ", "")
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("Resolution must look like WIDTH,HEIGHT (for example 640,480).")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("Resolution values must be positive integers.")
    return width, height


def open_video_source(
    camera_index: int = 0,
    fallback_video: Optional[str] = None,
    resolution: Optional[Tuple[int, int]] = None,
    prefer_camera: bool = True,
) -> Tuple[cv2.VideoCapture, str, str]:
    """Open camera first, optionally fallback to a video file.

    Returns:
        (capture, source_label, message)
    """
    if prefer_camera:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            if resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
            ok, _ = cap.read()
            if ok:
                return cap, "camera", f"Using camera index {camera_index}."
            cap.release()

    if fallback_video:
        fallback_path = Path(fallback_video)
        if fallback_path.exists():
            cap = cv2.VideoCapture(str(fallback_path))
            if cap.isOpened():
                return cap, "video", f"Camera unavailable. Using fallback video: {fallback_path}"

    raise RuntimeError(
        "Unable to open camera and no valid fallback video available. "
        "Check camera connection or pass --fallback-video <path>."
    )
