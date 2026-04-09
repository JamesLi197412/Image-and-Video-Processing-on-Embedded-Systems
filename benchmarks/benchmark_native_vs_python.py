from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mplconfig")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedded_utils.native_image_ops import ensure_native_shared_library
from lab05.edge_ops import laplace_filter, sobel_filter
from lab09.pasteImage import paste_image


@dataclass
class BenchmarkRow:
    operation: str
    input_shape: str
    correctness: str
    python_ms: float
    native_ms: float

    @property
    def speedup(self) -> float:
        return self.python_ms / self.native_ms if self.native_ms > 0 else float("inf")


def _median_ms(fn: Callable[[], object], iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()

    timings_ms: List[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        timings_ms.append(elapsed_ms)
    return statistics.median(timings_ms)


def _cpu_label() -> str:
    system = platform.system()
    if system == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            result = subprocess.run(
                ["sysctl", "-n", key],
                capture_output=True,
                text=True,
                check=False,
            )
            text = result.stdout.strip()
            if text:
                return text
    label = platform.processor() or platform.machine()
    return label if label else "unknown-cpu"


def _format_markdown_table(rows: List[BenchmarkRow]) -> str:
    lines = [
        "| Operation | Input | Correctness | Python median (ms) | Native C median (ms) | Speedup |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.operation} | {row.input_shape} | {row.correctness} | "
            f"{row.python_ms:.2f} | {row.native_ms:.2f} | {row.speedup:.2f}x |"
        )
    return "\n".join(lines)


def _benchmark_sobel(gray: np.ndarray, iterations: int, warmup: int) -> BenchmarkRow:
    py_result = sobel_filter(gray, backend="python")
    native_result = sobel_filter(gray, backend="c")
    max_abs_diff = float(np.max(np.abs(py_result[2] - native_result[2])))

    python_ms = _median_ms(lambda: sobel_filter(gray, backend="python"), iterations, warmup)
    native_ms = _median_ms(lambda: sobel_filter(gray, backend="c"), iterations, warmup)
    return BenchmarkRow(
        operation="Lab 05 Sobel magnitude",
        input_shape=f"{gray.shape[0]}x{gray.shape[1]} grayscale",
        correctness=f"max abs diff = {max_abs_diff:.4f}",
        python_ms=python_ms,
        native_ms=native_ms,
    )


def _benchmark_laplace(gray: np.ndarray, iterations: int, warmup: int) -> BenchmarkRow:
    py_result = laplace_filter(gray, backend="python")
    native_result = laplace_filter(gray, backend="c")
    max_abs_diff = float(np.max(np.abs(py_result - native_result)))

    python_ms = _median_ms(lambda: laplace_filter(gray, backend="python"), iterations, warmup)
    native_ms = _median_ms(lambda: laplace_filter(gray, backend="c"), iterations, warmup)
    return BenchmarkRow(
        operation="Lab 05 Laplace filter",
        input_shape=f"{gray.shape[0]}x{gray.shape[1]} grayscale",
        correctness=f"max abs diff = {max_abs_diff:.4f}",
        python_ms=python_ms,
        native_ms=native_ms,
    )


def _benchmark_alpha_blend(
    overlay: np.ndarray,
    background: np.ndarray,
    x: int,
    y: int,
    iterations: int,
    warmup: int,
) -> BenchmarkRow:
    py_result = paste_image(
        overlay,
        background.copy(),
        x=x,
        y=y,
        w=overlay.shape[1],
        h=overlay.shape[0],
        backend="python",
    )
    native_result = paste_image(
        overlay,
        background.copy(),
        x=x,
        y=y,
        w=overlay.shape[1],
        h=overlay.shape[0],
        backend="c",
    )
    max_abs_diff = int(np.max(np.abs(py_result.astype(np.int16) - native_result.astype(np.int16))))

    python_ms = _median_ms(
        lambda: paste_image(
            overlay,
            background.copy(),
            x=x,
            y=y,
            w=overlay.shape[1],
            h=overlay.shape[0],
            backend="python",
        ),
        iterations,
        warmup,
    )
    native_ms = _median_ms(
        lambda: paste_image(
            overlay,
            background.copy(),
            x=x,
            y=y,
            w=overlay.shape[1],
            h=overlay.shape[0],
            backend="c",
        ),
        iterations,
        warmup,
    )
    return BenchmarkRow(
        operation="Lab 09 alpha blend",
        input_shape=f"{overlay.shape[0]}x{overlay.shape[1]} over {background.shape[0]}x{background.shape[1]}",
        correctness=f"max channel diff = {max_abs_diff}",
        python_ms=python_ms,
        native_ms=native_ms,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Python/SciPy paths against the native C bridge.")
    parser.add_argument("--iterations", type=int, default=30, help="Timed iterations per benchmark.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per benchmark.")
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Optional path to write the markdown benchmark report.",
    )
    args = parser.parse_args()

    ensure_native_shared_library()

    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(1024, 1024), dtype=np.uint8)
    background = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)
    overlay = rng.integers(0, 256, size=(320, 320, 4), dtype=np.uint8)
    overlay[:, :, 3] = rng.integers(0, 256, size=(320, 320), dtype=np.uint8)

    x = 200
    y = 180
    rows = [
        _benchmark_sobel(gray, args.iterations, args.warmup),
        _benchmark_laplace(gray, args.iterations, args.warmup),
        _benchmark_alpha_blend(overlay, background, x, y, args.iterations, args.warmup),
    ]

    report_lines = [
        f"Benchmark date: {date.today().isoformat()}",
        f"Host: {_cpu_label()} ({platform.system()} {platform.release()}, {platform.machine()})",
        f"Python: {platform.python_version()}",
        "",
        _format_markdown_table(rows),
        "",
        "Notes:",
        "- Timings are host-side medians from the current machine, not Raspberry Pi measurements.",
        "- The native path measures the same Python-callable code path used by the updated lab helpers.",
    ]
    report = "\n".join(report_lines)
    print(report)

    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
