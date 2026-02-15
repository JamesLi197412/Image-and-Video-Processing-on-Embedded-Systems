from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import norm


def normal_cdf() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-4, 4, 50)
    y = np.array([norm.cdf(value) for value in x])
    return x, y


def convert_rgb_to_gray(file_name: str) -> np.ndarray:
    path = Path(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    return np.array(Image.open(path).convert("L"))


def histogram_equalize(img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(img_gray, np.ndarray) and img_gray.ndim == 2

    histogram, _ = np.histogram(img_gray.flatten(), 256, [0, 256])
    pdf = histogram.astype(float) / img_gray.size
    cdf = np.cumsum(pdf)

    cdf_normalized = (cdf * 255).astype(np.uint8)
    img_equal = cdf_normalized[img_gray]

    return img_equal.astype(img_gray.dtype), cdf_normalized


def generate_equalization_plot(input_path: str, output_path: str) -> None:
    img_before = convert_rgb_to_gray(input_path)
    img_after, cdf_normalized = histogram_equalize(img_before)
    x, y = normal_cdf()

    plt.figure(figsize=(15, 8))
    plt.subplot(231), plt.title("Before"), plt.axis("off")
    plt.imshow(img_before, cmap="gray", vmin=0, vmax=255)

    plt.subplot(232)
    plt.plot(x, y, linewidth=2)
    plt.title("Normal CDF"), plt.xlabel("x"), plt.ylabel("Probability")

    plt.subplot(233), plt.title("After")
    plt.imshow(img_after, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.hist(img_before.flatten(), bins=256, range=(0, 256), color="blue", edgecolor="none")
    plt.title("Histogram Original")
    plt.xlim(0, 260)

    plt.subplot(2, 3, 5)
    plt.plot(np.arange(256), cdf_normalized, linewidth=2)
    plt.title("Transformation")
    plt.xlim(0, 260)
    plt.ylim(0, 260)

    plt.subplot(2, 3, 6)
    plt.hist(img_after.flatten(), bins=256, range=(0, 256), color="blue", edgecolor="none")
    plt.title("Histogram Equalized")
    plt.xlim(0, 255)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply histogram equalization and generate summary plot.")
    parser.add_argument("--input", default="mandrill.png", help="Input image path.")
    parser.add_argument(
        "--output",
        default="Plot resulting from the histogram equalization.png",
        help="Output plot path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        generate_equalization_plot(args.input, args.output)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Saved equalization plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
