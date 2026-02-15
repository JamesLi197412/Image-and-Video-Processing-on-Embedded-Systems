from __future__ import annotations

import csv
import datetime as dt
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency fallback
    psutil = None


class RuntimeMetricsLogger:
    """Write per-frame runtime metrics to CSV for embedded pipeline profiling."""

    def __init__(
        self,
        csv_path: str,
        source: str,
        target_fps: Optional[float] = None,
        fps_window: int = 30,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.source = source
        self.target_fps = target_fps if target_fps and target_fps > 0 else None
        self.frame_times: Deque[float] = deque(maxlen=max(2, fps_window))
        self.total_dropped_frames = 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            return
        with self.csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_utc",
                    "frame_index",
                    "latency_ms",
                    "fps_window",
                    "cpu_percent",
                    "memory_percent",
                    "dropped_frames_frame",
                    "dropped_frames_total",
                    "source",
                    "note",
                ]
            )

    def begin_frame(self) -> float:
        return time.perf_counter()

    def log_frame(
        self,
        frame_index: int,
        started_at: float,
        note: str = "",
        dropped_frames: int = 0,
    ) -> None:
        now = time.perf_counter()
        latency_s = max(0.0, now - started_at)
        self.frame_times.append(now)

        if dropped_frames < 0:
            dropped_frames = 0
        self.total_dropped_frames += dropped_frames

        if self.target_fps and self.target_fps > 0:
            expected_interval = 1.0 / self.target_fps
            late_frames = int(latency_s / expected_interval) - 1
            if late_frames > 0:
                self.total_dropped_frames += late_frames
                dropped_frames += late_frames

        fps_window = self._compute_fps()
        cpu_percent = psutil.cpu_percent(interval=None) if psutil else ""
        mem_percent = psutil.virtual_memory().percent if psutil else ""

        with self.csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                    frame_index,
                    round(latency_s * 1000.0, 3),
                    round(fps_window, 3),
                    cpu_percent,
                    mem_percent,
                    dropped_frames,
                    self.total_dropped_frames,
                    self.source,
                    note,
                ]
            )

    def _compute_fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / elapsed
