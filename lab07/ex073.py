import cv2
import numpy as np
import math

gray = cv2.imread("mandrill.png", 0)

scale_levels = 5
sigma = 1.6
k = math.sqrt(2)

# ---------------------------------------------------------
# 1. First octave – Gaussian scale space
# ---------------------------------------------------------
scale_space_gauss = np.zeros(
    (gray.shape[0], gray.shape[1], scale_levels),
    dtype=np.uint8
)

for i in range(scale_levels):
    k_sigma = sigma * (k ** i)
    scale_space_gauss[:, :, i] = cv2.GaussianBlur(
        gray,
        ksize=(0, 0),
        sigmaX=k_sigma,
        sigmaY=k_sigma
    )
    cv2.imwrite(f"imgGauss_Oct1_{i}.png", scale_space_gauss[:, :, i])


# ---------------------------------------------------------
# 2. Difference of Gaussians for octave 1
# ---------------------------------------------------------
scale_space_dog = np.zeros(
    (gray.shape[0], gray.shape[1], scale_levels - 1),
    dtype=np.int16     # allow negative values
)

for i in range(scale_levels - 1):
    scale_space_dog[:, :, i] = (
        scale_space_gauss[:, :, i + 1].astype(np.int16)
        - scale_space_gauss[:, :, i].astype(np.int16)
    )
    cv2.imwrite(f"imgDoG_Oct1_{i+1}.png", scale_space_dog[:, :, i])


# ---------------------------------------------------------
# 3. Second octave: downsample original image
# ---------------------------------------------------------
next_oct_gray = cv2.pyrDown(gray)

scale_space_gauss_2 = np.zeros(
    (next_oct_gray.shape[0], next_oct_gray.shape[1], scale_levels),
    dtype=np.uint8
)

for i in range(scale_levels):
    k_sigma = sigma * (k ** i)
    scale_space_gauss_2[:, :, i] = cv2.GaussianBlur(
        next_oct_gray,
        ksize=(0, 0),
        sigmaX=k_sigma,
        sigmaY=k_sigma
    )
    cv2.imwrite(f"imgGauss_Oct2_{i}.png", scale_space_gauss_2[:, :, i])


# ---------------------------------------------------------
# 4. DoG for octave 2
# ---------------------------------------------------------
scale_space_dog_2 = np.zeros(
    (next_oct_gray.shape[0], next_oct_gray.shape[1], scale_levels - 1),
    dtype=np.int16
)

for i in range(scale_levels - 1):
    scale_space_dog_2[:, :, i] = (
        scale_space_gauss_2[:, :, i + 1].astype(np.int16)
        - scale_space_gauss_2[:, :, i].astype(np.int16)
    )
    cv2.imwrite(f"imgDoG_Oct2_{i+1}.png", scale_space_dog_2[:, :, i])
