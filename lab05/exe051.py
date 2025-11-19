# Exercise 5.1
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
import numpy as np
import cv2
import scipy.ndimage as ndim
from typing import Tuple

def edge_detector_id(np_img: np.ndarray, dm: np.ndarray, dn:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert all(isinstance (arr, np.ndarray) for arr in (np_img, dm, dn))
    fm = ndim.correlate1d(np_img, dm, axis = 1)
    fn = ndim.correlate1d(np_img, dn, axis = 0)
    magnitued = np.sqrt(fm**2 + fn**2)
    phase = np.arctan2(fm, fn)
    return fm, fn, magnitued, phase

def image_visual(filename:str, output:str) :
    img = Image.open(filename).convert('L')
    gray_img = np.array(img, "uint8")

    dm = np.array([[0,0,0], [-1, 1, 0], [0,0,0]])
    dn = np.array([[0,-1,0], [0, 1, 0], [0,0,0]])

    dm_symmetrical = np.array([[0,0,0],[-1,0,1],[0,0,0]])
    dn_symmetrical = np.array([[0,-1,0],[0,0,0],[0,1,0]])

    dm_forward = np.array([[0,0,0],[0,-1,1],[0,0,0]])
    dn_forward = np.array([[0,0,0],[0,-1,0],[0,1,0]])

    # Backend Gradient
    dm_backward = np.array([-1, 1, 0])
    dn_backward = np.array([-1, 1, 0])

    # Symmetrical Gradient (Eqs. 5.13 and 5.14)
    dm_symmetrical_1d = np.array([-1, 0, 1])
    dn_symmetrical_1d = np.array([-1, 0, 1])

    #  Forward Gradient (Eqs. 5.15 and 5.16)
    dm_forward_1d = np.array([0, -1, 1])
    dn_forward_1d = np.array([0, -1, 1])


    fm1, fn1, magnitude1, phase1 = edge_detector_id(gray_img, dm_backward, dn_backward)

    fm2, fn2, magnitude2, phase2 = edge_detector_id(gray_img, dm_symmetrical_1d, dn_symmetrical_1d)

    fm3, fn3, magnitude3, phase3 = edge_detector_id(gray_img,  dm_forward_1d, dn_forward_1d)


    plt.subplot(5,3,1), plt.axis("off")
    plt.imshow(gray_img, cmap='gray'), plt.title("Gradient 1")
    plt.subplot(5,3,2), plt.axis("off")
    plt.imshow(gray_img, cmap='gray'), plt.title("Gradient 2")
    plt.subplot(5,3,3), plt.axis("off")
    plt.imshow(gray_img, cmap='gray'), plt.title("Gradient 3")

    plt.subplot(5, 3, 4), plt.axis("off")
    plt.imshow(fm1, "gray")
    plt.subplot(5, 3, 5), plt.axis("off")
    plt.imshow(fm2, "gray"), plt.title("Fm vertical edges")
    plt.subplot(5, 3, 6), plt.axis("off")
    plt.imshow(fm3, "gray")

    plt.subplot(5, 3, 7), plt.axis("off")
    plt.imshow(fn1, "gray")
    plt.subplot(5, 3, 8), plt.axis("off")
    plt.imshow(fn2, "gray"), plt.title("Fn horizont edges")
    plt.subplot(5, 3, 9), plt.axis("off")
    plt.imshow(fn3, "gray")

    plt.subplot(5, 3, 10), plt.axis("off")
    plt.imshow(magnitude1, "gray")
    plt.subplot(5, 3, 11), plt.axis("off")
    plt.imshow(magnitude2, "gray"), plt.title("Absolute Value")
    plt.subplot(5, 3, 12), plt.axis("off")
    plt.imshow(magnitude3, "gray")

    plt.subplot(5, 3, 13), plt.axis("off")
    plt.imshow(phase1, "gray")
    plt.subplot(5, 3, 14), plt.axis("off")
    plt.imshow(phase2, "gray"), plt.title("Phase")
    plt.subplot(5, 3, 15), plt.axis("off")
    plt.imshow(phase3, "gray")

    plt.tight_layout()
    plt.savefig(output)


if __name__ == "__main__":
    image_visual("chessboard.pbm", "filtered.png")

    # Exercise 5.2
    image_visual("chessboard_crooked.png", "filtered_crooked.png")
