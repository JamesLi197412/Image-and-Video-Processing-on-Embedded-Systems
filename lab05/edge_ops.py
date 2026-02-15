from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.ndimage as ndim


Array = np.ndarray


def load_gray_image(path: str) -> Array:
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    return np.array(Image.open(img_path).convert("L"), dtype=np.uint8)


def edge_detector_1d(np_img: Array, dm: Array, dn: Array) -> Tuple[Array, Array, Array, Array]:
    np_img = np_img.astype(float)
    fm = ndim.correlate1d(np_img, dm, axis=1)
    fn = ndim.correlate1d(np_img, dn, axis=0)
    magnitude = np.sqrt(fm ** 2 + fn ** 2)
    phase = np.arctan2(fm, fn)
    return fm, fn, magnitude, phase


def edge_detector_2d(np_img: Array, dm: Array, dn: Array) -> Tuple[Array, Array, Array, Array]:
    np_img = np_img.astype(float)
    fm = ndim.correlate(np_img, dm)
    fn = ndim.correlate(np_img, dn)
    magnitude = np.sqrt(fm ** 2 + fn ** 2)
    phase = np.arctan2(fm, fn)
    return fm, fn, magnitude, phase


def sobel_filter(np_img: Array) -> Tuple[Array, Array, Array, Array]:
    np_img = np_img.astype(float)
    fm = ndim.sobel(np_img, axis=1)
    fn = ndim.sobel(np_img, axis=0)
    magnitude = np.sqrt(fm ** 2 + fn ** 2)
    phase = np.arctan2(fm, fn)
    return fm, fn, magnitude, phase


def gradient_variants(gray_img: Array) -> Dict[str, Tuple[Array, Array, Array, Array]]:
    kernels = {
        "backward": (np.array([-1, 1, 0]), np.array([-1, 1, 0])),
        "symmetric": (np.array([-1, 0, 1]), np.array([-1, 0, 1])),
        "forward": (np.array([0, -1, 1]), np.array([0, -1, 1])),
    }
    return {name: edge_detector_1d(gray_img, dm, dn) for name, (dm, dn) in kernels.items()}


def plot_gradient_comparison(gray_img: Array, output_path: str) -> None:
    variants = gradient_variants(gray_img)
    order = ["backward", "symmetric", "forward"]

    plt.figure(figsize=(12, 12))
    for i, name in enumerate(order, start=1):
        fm, fn, magnitude, phase = variants[name]

        plt.subplot(4, 3, i)
        plt.imshow(gray_img, cmap="gray")
        plt.axis("off")
        plt.title(f"{name} - input")

        plt.subplot(4, 3, i + 3)
        plt.imshow(fm, cmap="gray")
        plt.axis("off")
        plt.title("Fm")

        plt.subplot(4, 3, i + 6)
        plt.imshow(fn, cmap="gray")
        plt.axis("off")
        plt.title("Fn")

        plt.subplot(4, 3, i + 9)
        plt.imshow(magnitude, cmap="gray")
        plt.axis("off")
        plt.title("Magnitude")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sobel_comparison(gray_img: Array, output_path: str) -> None:
    dm = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    dn = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    ddm = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    ddn = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])

    fm1, fn1, magnitude1, phase1 = sobel_filter(gray_img)
    fm2, fn2, magnitude2, phase2 = edge_detector_2d(gray_img, dm, dn)
    fm3, fn3, magnitude3, phase3 = edge_detector_2d(gray_img, ddm, ddn)

    plt.figure(figsize=(12, 14))
    images = [
        (gray_img, "Input"),
        (gray_img, "Input"),
        (gray_img, "Input"),
        (fm1, "Fm1"),
        (fm2, "Fm2"),
        (fm3, "Fm3"),
        (fn1, "Fn1"),
        (fn2, "Fn2"),
        (fn3, "Fn3"),
        (magnitude1, "Magnitude1"),
        (magnitude2, "Magnitude2"),
        (magnitude3, "Magnitude3"),
        (phase1, "Phase1"),
        (phase2, "Phase2"),
        (phase3, "Phase3"),
    ]

    for idx, (img, title) in enumerate(images, start=1):
        plt.subplot(5, 3, idx)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def gauss_filter(np_img: Array, sigma: float) -> Tuple[Array, Array]:
    fm = ndim.gaussian_filter(np_img, (sigma, sigma), (0, 1))
    fn = ndim.gaussian_filter(np_img, (sigma, sigma), (1, 0))
    return fm, fn


def laplace_filter(np_img: Array) -> Array:
    return ndim.laplace(np_img)


def laplace_gauss_filter(np_img: Array, sigma: float) -> Array:
    return ndim.gaussian_laplace(np_img, sigma)


def plot_filter_family(gray_img: Array, output_path: str, sigma: float = 5.0) -> None:
    fm1, fn1, _, _ = sobel_filter(gray_img)
    fm2, fn2 = gauss_filter(gray_img, sigma)
    fmn1 = laplace_filter(gray_img)
    fmn2 = laplace_gauss_filter(gray_img, sigma)

    plt.figure(figsize=(12, 10))
    images = [
        (fm1, "Sobel m"),
        (fn1, "Sobel n"),
        (fm2, "Gauss m"),
        (fn2, "Gauss n"),
        (fmn1, "Laplace mn"),
        (fmn2, "Laplace Gauss mn"),
    ]

    for idx, (img, title) in enumerate(images, start=1):
        plt.subplot(3, 2, idx)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(title)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sigma_variation(gray_img: Array, output_path: str, sigma_values: List[int]) -> None:
    plt.figure(figsize=(16, 12))
    for idx, sig in enumerate(sigma_values):
        fm_smooth = ndim.gaussian_filter(gray_img, (sig, sig), (0, 0))
        plt.subplot(2, 2, idx + 1)
        plt.imshow(fm_smooth, cmap="gray")
        plt.axis("off")
        plt.title(f"Gaussian smoothing σ={sig}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_gaussian_derivatives(gray_img: Array, output_path: str, sigma_values: List[int]) -> None:
    plt.figure(figsize=(16, 12))
    for idx, sig in enumerate(sigma_values):
        fm_deriv = ndim.gaussian_filter(gray_img, (sig, sig), (0, 1))
        fn_deriv = ndim.gaussian_filter(gray_img, (sig, sig), (1, 0))
        magnitude = np.sqrt(fm_deriv ** 2 + fn_deriv ** 2)

        plt.subplot(2, 2, idx + 1)
        plt.imshow(magnitude, cmap="gray")
        plt.axis("off")
        plt.title(f"Gaussian gradient magnitude σ={sig}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
