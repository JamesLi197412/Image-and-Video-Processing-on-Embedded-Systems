import cv2
import numpy as np


def add_alpha_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("add_alpha_channel expects a BGR image (H, W, 3).")

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    b_channel, g_channel, r_chanel =  cv2.split(img[:,:,3])

    black_mask = (b_channel == 0) & (g_channel == 0) & (r_chanel == 0)
    bgra[black_mask, 3] = 0

    return bgra


