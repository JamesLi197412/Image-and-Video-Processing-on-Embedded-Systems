# Exercise 5.3
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import cv2
from pylab import *
from PIL import Image
import scipy.ndimage as ndim
from typing import Tuple
import os


def sobel_filter(np_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert isinstance(np_img, np.ndarray)
    np_img = np_img.astype(float)
    fm = ndim.sobel(np_img, axis=1)  # Horizontal edges
    fn = ndim.sobel(np_img, axis=0)  # Vertical edges
    magnitude = np.sqrt(fm**2 + fn**2)

    phase = np.arctan2(fm, fn)
    return fm, fn, magnitude, phase


def edge_detector(np_img: np.ndarray, dm: np.ndarray, dn: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert all(isinstance(arr, np.ndarray) for arr in (np_img, dm, dn))
    np_img = np_img.astype(float)
    fm = ndim.correlate(np_img, dm)
    fn = ndim.correlate(np_img, dn)

    magnitude = np.sqrt(fm**2 + fn**2)

    phase = np.arctan2(fm, fn)
    return fm, fn, magnitude, phase




# Define Sobel operator kernels according to Eqs. (5.17) and (5.18)
dm = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

dn = np.array([[-1, -2, -1],
               [0, 0, 0],
               [1, 2, 1]])

# Diagonal Sobel kernels according to Eqs. (5.19) and (5.20)
ddm = np.array([[0, 1, 2],
                [-1, 0, 1],
                [-2, -1, 0]])

ddn = np.array([[-2, -1, 0],
                [-1, 0, 1],
                [0, 1, 2]])

# Load the chessboard image
img = Image.open("chessboard_crooked.png").convert('L')
gray_img = np.array(img, "uint8")

# Apply Sobel filter using SciPy's built-in function
fm1, fn1, magnitude, phase = sobel_filter(gray_img)

# Apply manual Sobel implementation
fm2, fn2, magnitude2, phase2 = edge_detector(gray_img, dm, dn)

# Apply diagonal Sobel operators
fm3, fn3, magnitude3, phase3 = edge_detector(gray_img, ddm, ddn)

# Create the plot with 5 rows and 3 columns
plt.subplot(5, 3, 1), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Sobel 1")
plt.subplot(5, 3, 2), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Sobel 2")
plt.subplot(5, 3, 3), plt.axis("off")
plt.imshow(gray_img, "gray"), plt.title("Sobel 3")

# Display Fm (horizontal gradients)
plt.subplot(5, 3, 4), plt.axis("off")
plt.imshow(fm1, "gray"), plt.title("Fm1")
plt.subplot(5, 3, 5), plt.axis("off")
plt.imshow(fm2, "gray"), plt.title("Fm2")
plt.subplot(5, 3, 6), plt.axis("off")
plt.imshow(fm3, "gray"), plt.title("Fm3")

# Display Fn (vertical gradients)
plt.subplot(5, 3, 7), plt.axis("off")
plt.imshow(fn1, "gray"), plt.title("Fn1")
plt.subplot(5, 3, 8), plt.axis("off")
plt.imshow(fn2, "gray"), plt.title("Fn2")
plt.subplot(5, 3, 9), plt.axis("off")
plt.imshow(fn3, "gray"), plt.title("Fn3")

# Display magnitudes
plt.subplot(5, 3, 10), plt.axis("off")
plt.imshow(magnitude, "gray"), plt.title("Magnitude1")
plt.subplot(5, 3, 11), plt.axis("off")
plt.imshow(magnitude2, "gray"), plt.title("Magnitude2")
plt.subplot(5, 3, 12), plt.axis("off")
plt.imshow(magnitude3, "gray"), plt.title("Magnitude3")

# Display phases
plt.subplot(5, 3, 13), plt.axis("off")
plt.imshow(phase, "gray"), plt.title("Phase1")
plt.subplot(5, 3, 14), plt.axis("off")
plt.imshow(phase2, "gray"), plt.title("Phase2")
plt.subplot(5, 3, 15), plt.axis("off")
plt.imshow(phase3, "gray"), plt.title("Phase3")

plt.tight_layout()
plt.savefig("sobel_filtered.png")