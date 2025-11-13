import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
from scipy.stats import norm
import numpy as np
from typing import Tuple


def normal_cdf(x) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-4,4,50)
    y = [norm.cdf(x[i]) for i in range(x.size)]
    return x, y

def convert_rgb_to_gray(file_name: str) -> np.ndarray:
    assert isinstance(file_name, str)
    return np.array(Image.open(file_name).convert('L'))

def histogram_equalize(img_gray: np.ndarry) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(img_gray, np.ndarray) and img_gray.ndim == 2

    histogram, _ = np.histogram(img_gray.flatten(), 256, [0, 256])

    pdf = histogram.astype(float) / img_gray.size

    cdf = np.cumsum(pdf)

    cdf_normalized = (cdf * 255).astype(np.uint8)
    img_equal = cdf_normalized[img_gray]
    return img_equal.astype(img_gray.dtype), cdf

def main():
    mandrill_1 = convert_rgb_to_gray("mandrill.png")
    mandrill_2, cdf = histogram_equalize(mandrill_1)

    x,y = normal_cdf()

    # Calculate histograms for visualization
    hist_original, _ = np.histogram(mandrill_1.flatten(), bins=256, range=(0, 255))
    hist_equalized, _ = np.histogram(mandrill_2.flatten(), bins=256, range=(0, 255))

    # Normalize histograms to probabilities
    pdf_original = hist_original / mandrill_1.size
    pdf_equalized = hist_equalized / mandrill_2.size

    # Calculate CDF of equalized image
    cdf_equalized = np.cumsum(pdf_equalized)

    plt.figure(figsize=(10,8))
    plt.subplot(231), plt.title("Before"), plt.axis('off')
    plt.imshow(mandrill_1, cmap='gray', vmin= 0, vmax = 255)

    plt.subplot(232), plt.title("After"), plt.axis('off')
    plt.imshow()
