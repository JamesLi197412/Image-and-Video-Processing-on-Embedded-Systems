from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image_to_array(file_name: str) -> np.ndarray:
    path = Path(file_name)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def save_array_to_gray_img(np_array_img: np.ndarray, file_name: str) -> None:
    assert isinstance(np_array_img, np.ndarray), "It has to be a Numpy array"
    assert isinstance(file_name, str), "It has to be a string"
    assert np_array_img.dtype == np.uint8, f"Array dtype must be uint8, got {np_array_img.dtype}"

    img = Image.fromarray(np_array_img)
    img.save(file_name)


def convert_rgb_equal(img_rgb: np.ndarray) -> np.ndarray:
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    luminance = (r + g + b) / 3
    return luminance.astype(np.uint8)


def convert_rgb_weighted_1(img_rgb: np.ndarray) -> np.ndarray:
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance.astype(np.uint8)


def convert_rgb_weighted_2(img_rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114])
    luminance = np.dot(img_rgb, weights)
    return luminance.astype(np.uint8)


# User-defined histogram function
def hist_self(img_gray: np.ndarray, bins: int = 256) -> List[int]:
    hist = [0] * bins
    for pixel in img_gray.flatten():
        hist[int(pixel)] += 1
    return hist


def plot_histogram(img_gray: np.ndarray, plot_path: str) -> None:
    """Plot histogram of a grayscale image and save it."""
    assert isinstance(img_gray, np.ndarray), "img_gray must be a NumPy array"
    assert img_gray.dtype == np.uint8, f"Image must be uint8, got {img_gray.dtype}"
    assert len(img_gray.shape) == 2, f"Image must be grayscale (2D), got shape {img_gray.shape}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    pixel_values = img_gray.flatten()
    counts = hist_self(img_gray)
    ax1.plot(range(256), counts, color="gray", linewidth=2)
    ax1.set_title("Histogram (manual)")
    ax1.set_xlabel("Intensity Value")
    ax1.set_ylabel("Frequency")
    ax1.grid(True, alpha=0.3)

    ax2.hist(pixel_values, bins=256, range=(0, 255), color="blue", edgecolor="black", alpha=0.7)
    ax2.set_title("Histogram (matplotlib)")
    ax2.set_xlabel("Intensity Value")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved histogram to {plot_path}")
    plt.close()


def comparison(filename: str, output_plot: str) -> None:
    mandrill = load_image_to_array(filename)
    mandrill_gray = convert_rgb_weighted_2(mandrill)
    plot_histogram(mandrill_gray, output_plot)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot grayscale histogram.")
    parser.add_argument("--input", default="mandrill.png", help="Input RGB image path.")
    parser.add_argument(
        "--output",
        default="Histogram for mandrill.png",
        help="Output plot file path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        comparison(args.input, args.output)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
