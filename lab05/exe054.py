# Exercise 5.4
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import cv2
from pylab import *
from PIL import Image
import scipy.ndimage as ndim
from typing import Tuple, Union, List
import os
from exe053 import sobel_filter

sigma_values = [1, 3, 7, 10]
gray_img = np.array(
    Image.open( "chessboard_crooked.png").convert("L"),
    "uint8",
)


def gauss_filter(np_img: np.ndarray, sigma: Union[int, float]) -> Tuple[np.ndarray, np.ndarray]:
    # sigma: width of Gaussian filter mask
    assert isinstance(np_img, np.ndarray) and isinstance(sigma, int | float)
    # order=1 means first derivative, (0,1) means derivative in x-direction
    fm = ndim.gaussian_filter(np_img, (sigma, sigma), (0, 1))  # x-direction
    fn = ndim.gaussian_filter(np_img, (sigma, sigma), (1, 0)) # y-direction
    return fm, fn


def laplace_filter(np_img: np.ndarray) -> np.ndarray:
    assert isinstance(np_img, np.ndarray)
    # Apply Laplacian filter (second derivative)
    fmn = ndim.laplace(np_img)
    return fmn


def laplace_gauss_filter(np_img: np.ndarray, sigma: Union[int, float]) -> np.ndarray:
    assert isinstance(np_img, np.ndarray) and isinstance(sigma, int | float)
    # Apply Laplacian of Gaussian (LoG) filter
    fmn = ndim.gaussian_laplace(np_img, sigma)
    return fmn


def main_operation(gray_img: np.ndarray, sigma: Union[int, float] =5):
    # Standard deviation for Gaussian filters
    sigma = 5

    # Apply all filters
    fm1, fn1, _, _ = sobel_filter(gray_img)
    fm2, fn2 = gauss_filter(gray_img, sigma)
    fmn1 = laplace_filter(gray_img)
    fmn2 = laplace_gauss_filter(gray_img, sigma)

    # Create the main comparison plot (3x2 grid)
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1), plt.axis("off")
    plt.imshow(fm1, "gray"), plt.title("Sobel m")
    plt.subplot(3, 2, 2), plt.axis("off")
    plt.imshow(fn1, "gray"), plt.title("Sobel n")
    plt.subplot(3, 2, 3), plt.axis("off")
    plt.imshow(fm2, "gray"), plt.title("Gauss m")
    plt.subplot(3, 2, 4), plt.axis("off")
    plt.imshow(fn2, "gray"), plt.title("Gauss n")
    plt.subplot(3, 2, 5), plt.axis("off")
    plt.imshow(fmn1, "gray"), plt.title("Laplace mn")
    plt.subplot(3, 2, 6), plt.axis("off")
    plt.imshow(fmn2, "gray"), plt.title("Laplace Gauss mn")

    plt.tight_layout()
    plt.savefig("laplace_filtered.png")
    plt.close()

main_operation(gray_img, 5)


# Task 2: Test different sigma values with order (0,0)
def sigma_variation(gray_img:np.ndarray):
    sigma_values = [1, 3, 7, 10]

    plt.figure(figsize=(16, 12))
    for idx, sig in enumerate(sigma_values):
        # Apply Gaussian filter with order (0,0) - just smoothing, no derivative
        fm_smooth = ndim.gaussian_filter(gray_img, (sig, sig), (0, 0))
        fn_smooth = ndim.gaussian_filter(gray_img, (sig, sig), (0, 0))

        magnitude = np.sqrt(fm_smooth ** 2 + fn_smooth ** 2)

        plt.subplot(2, 2, idx + 1), plt.axis("off")
        plt.imshow(fm_smooth, "gray")
        plt.title(f"Gaussian smoothing σ={sig}")

    plt.tight_layout()
    plt.savefig("gaussian_smoothing_comparison.png")
    plt.close()

sigma_variation(gray_img)


# Task 3: Test Gaussian derivatives with different sigma values
def gaussian_derivatives(gray_img:np.ndarray, sigma_values:List[int]):
    #sigma_values = [1, 3, 7, 10]
    plt.figure(figsize=(16, 12))
    for idx, sig in enumerate(sigma_values):
        fm_deriv = ndim.gaussian_filter(gray_img, (sig, sig), (0, 1))
        fn_deriv = ndim.gaussian_filter(gray_img, (sig, sig), (1, 0))
        magnitude = np.sqrt(fm_deriv ** 2 + fn_deriv ** 2)

        plt.subplot(2, 2, idx + 1), plt.axis("off")
        plt.imshow(magnitude, "gray")
        plt.title(f"Gaussian gradient magnitude σ={sig}")

    plt.tight_layout()
    plt.savefig("gaussian_gradient_comparison.png")
    plt.close()


gaussian_derivatives(gray_img, sigma_values)

