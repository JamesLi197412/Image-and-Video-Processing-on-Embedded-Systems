import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
import numpy as np
from typing import Tuple

def normal_cdf() -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-4,4,50)
    y = [norm.cdf(x[i]) for i in range(x.size)]
    return x, y

def convert_rgb_to_gray(file_name: str) -> np.ndarray:
    assert isinstance(file_name, str)
    return np.array(Image.open(file_name).convert('L'))

def histogram_equalize(img_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(img_gray, np.ndarray) and img_gray.ndim == 2

    histogram, _ = np.histogram(img_gray.flatten(), 256, [0, 256])

    pdf = histogram.astype(float) / img_gray.size

    cdf = np.cumsum(pdf)

    cdf_normalized = (cdf * 255).astype(np.uint8)
    img_equal = cdf_normalized[img_gray]

    return img_equal.astype(img_gray.dtype), cdf_normalized

def main():
    mandrill_1 = convert_rgb_to_gray("mandrill.png")
    mandrill_2, cdf_normalized = histogram_equalize(mandrill_1)

    x,y = normal_cdf()

    # Calculate histograms for visualization
    plt.figure(figsize=(15,8))
    plt.subplot(231), plt.title("Before"), plt.axis('off')
    plt.imshow(mandrill_1, cmap='gray', vmin= 0, vmax = 255)

    plt.subplot(232)
    plt.plot(x,y, linewidth=2)
    plt.title("Normal cdf"), plt.xlabel("x"), plt.ylabel("Probability")

    plt.subplot(233), plt.title("After")
    plt.imshow(mandrill_2, cmap='gray', vmin= 0, vmax = 255)
    plt.axis("off")

    # Histogram Original
    plt.subplot(2,3,4)
    plt.hist(mandrill_1.flatten(), bins=256, range=(0, 256), color='blue', edgecolor='none')
    plt.title("Histogram Original")
    plt.xlim(0,260)


    plt.subplot(2,3,5)
    plt.plot(np.arange(256), cdf_normalized, linewidth=2)
    plt.title("Transformation")
    plt.xlim(0,260)
    plt.ylim(0,260)

    plt.subplot(2,3,6)
    plt.hist(mandrill_2.flatten(), bins=256, range=(0, 256), color='blue', edgecolor='none')
    plt.title("Histogram CDF-Normalized")
    plt.xlim(0,255)

    plt.tight_layout()
    plt.savefig("Plot resulting from the histogram equalization.png", dpi = 150)
    plt.close()

main()