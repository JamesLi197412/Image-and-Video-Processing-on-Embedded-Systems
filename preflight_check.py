#!/usr/bin/env python3
from __future__ import annotations

import argparse
import grp
import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


def _status(kind: str, message: str) -> None:
    print(f"[{kind}] {message}")


def check_imports(modules: Iterable[str]) -> Tuple[int, int]:
    ok = 0
    fail = 0
    for name in modules:
        try:
            importlib.import_module(name)
            _status("OK", f"Python module available: {name}")
            ok += 1
        except Exception as exc:
            _status("FAIL", f"Python module missing: {name} ({exc})")
            fail += 1
    return ok, fail


def check_files(paths: Iterable[str], label: str) -> Tuple[int, int]:
    ok = 0
    fail = 0
    for file_path in paths:
        p = Path(file_path)
        if p.exists():
            _status("OK", f"{label} found: {p}")
            ok += 1
        else:
            _status("FAIL", f"{label} missing: {p}")
            fail += 1
    return ok, fail


def check_groups(required_groups: Iterable[str]) -> Tuple[int, int]:
    ok = 0
    fail = 0
    user_groups = {grp.getgrgid(gid).gr_name for gid in os.getgroups()}
    for group_name in required_groups:
        if group_name in user_groups:
            _status("OK", f"Current user belongs to group '{group_name}'")
            ok += 1
        else:
            _status(
                "FAIL",
                (
                    f"Current user is not in group '{group_name}'. "
                    f"Run: sudo usermod -aG {group_name} $USER, then re-login."
                ),
            )
            fail += 1
    return ok, fail


def check_camera(camera_index: int) -> Tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        _status("FAIL", f"Camera index {camera_index} could not be opened by OpenCV.")
        return 0, 1
    ok, _frame = cap.read()
    cap.release()
    if ok:
        _status("OK", f"Camera index {camera_index} opened and returned a frame.")
        return 1, 0
    _status("FAIL", f"Camera index {camera_index} opened but returned no frames.")
    return 0, 1


def check_video_readable(video_path: str) -> Tuple[int, int]:
    import cv2

    p = Path(video_path)
    if not p.exists():
        _status("FAIL", f"Fallback video not found: {p}")
        return 0, 1
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        _status("FAIL", f"Fallback video cannot be opened: {p}")
        return 0, 1
    ok, _ = cap.read()
    cap.release()
    if ok:
        _status("OK", f"Fallback video readable: {p}")
        return 1, 0
    _status("FAIL", f"Fallback video opened but no frames: {p}")
    return 0, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight checks for embedded vision labs.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index to validate.")
    parser.add_argument("--require-camera", action="store_true", help="Fail if camera cannot be opened.")
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Video file path that must exist (can be repeated).",
    )
    parser.add_argument(
        "--cascade",
        action="append",
        default=[],
        help="Cascade XML path that must exist (can be repeated).",
    )
    parser.add_argument(
        "--check-fallback-video",
        default="",
        help="Optional fallback video path to validate by opening and reading one frame.",
    )
    parser.add_argument(
        "--skip-gpio-group-check",
        action="store_true",
        help="Skip Linux group membership checks for video/gpio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    _status("INFO", f"Python version: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        _status("FAIL", "Python 3.10+ is required.")
        return 1

    total_ok = 0
    total_fail = 0

    # Core requirements for these labs.
    mod_ok, mod_fail = check_imports(
        ["numpy", "PIL", "matplotlib", "scipy", "cv2", "psutil"]
    )
    total_ok += mod_ok
    total_fail += mod_fail

    # Optional Raspberry Pi runtime requirements.
    rpi_ok, rpi_fail = check_imports(["gpiozero", "picamera2"])
    total_ok += rpi_ok
    total_fail += rpi_fail

    file_ok, file_fail = check_files(args.cascade, "Cascade file")
    total_ok += file_ok
    total_fail += file_fail

    video_ok, video_fail = check_files(args.video, "Video file")
    total_ok += video_ok
    total_fail += video_fail

    if args.check_fallback_video:
        ok, fail = check_video_readable(args.check_fallback_video)
        total_ok += ok
        total_fail += fail

    if not args.skip_gpio_group_check:
        grp_ok, grp_fail = check_groups(["video", "gpio"])
        total_ok += grp_ok
        total_fail += grp_fail

    if args.require_camera:
        cam_ok, cam_fail = check_camera(args.camera_index)
        total_ok += cam_ok
        total_fail += cam_fail

    _status("INFO", f"Checks completed: {total_ok} OK, {total_fail} FAIL")
    return 1 if total_fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
