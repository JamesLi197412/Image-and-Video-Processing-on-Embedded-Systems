from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def build_scale_space(gray: np.ndarray, scale_levels: int, sigma: float, octave_prefix: str) -> np.ndarray:
    k = math.sqrt(2)
    scale_space_gauss = np.zeros((gray.shape[0], gray.shape[1], scale_levels), dtype=np.uint8)

    for i in range(scale_levels):
        k_sigma = sigma * (k ** i)
        scale_space_gauss[:, :, i] = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=k_sigma, sigmaY=k_sigma)
        cv2.imwrite(f"imgGauss_{octave_prefix}_{i}.png", scale_space_gauss[:, :, i])

    scale_space_dog = np.zeros((gray.shape[0], gray.shape[1], scale_levels - 1), dtype=np.int16)
    for i in range(scale_levels - 1):
        scale_space_dog[:, :, i] = scale_space_gauss[:, :, i + 1].astype(np.int16) - scale_space_gauss[:, :, i].astype(np.int16)
        cv2.imwrite(f"imgDoG_{octave_prefix}_{i+1}.png", scale_space_dog[:, :, i])

    return scale_space_gauss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gaussian/DoG scale space across two octaves.")
    parser.add_argument("--input", default="mandrill.png", help="Input grayscale image path.")
    parser.add_argument("--levels", type=int, default=5, help="Number of scale levels per octave.")
    parser.add_argument("--sigma", type=float, default=1.6, help="Base sigma.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        print(f"Error: input image not found: {path}")
        return 1

    gray = cv2.imread(str(path), 0)
    if gray is None:
        print(f"Error: cannot decode image: {path}")
        return 1

    build_scale_space(gray, args.levels, args.sigma, octave_prefix="Oct1")
    next_oct_gray = cv2.pyrDown(gray)
    build_scale_space(next_oct_gray, args.levels, args.sigma, octave_prefix="Oct2")

    print("Generated Gaussian and DoG images for octave 1 and octave 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
