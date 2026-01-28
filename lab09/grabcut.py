import numpy as np
import cv2
import time

# reference: https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/
def grab_cut(frame: np.ndarray, x: int, y: int, w: int, h: int, iter_count: int = 5) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    #
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    rect = (x, y, w, h)

    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype("uint8")
    result = frame * mask2[:, :, np.newaxis]
    return result


