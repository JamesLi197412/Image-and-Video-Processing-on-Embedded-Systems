import time
from typing import Tuple
import cv2
# pip install picamera2 -- run in the linux environment
from picamera2 import Picamera2
import numpy as np

def take_img(res:[Tuple[int,int]]) -> np.ndarray:
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main = {"format": "XRGB8888", "size": res},
        )
    )
    picam2.start()
    time.sleep(2)
    bgr_frame = picam2.capture_array()
    picam2.stop()
    return bgr_frame

res = (640, 480)
img = take_img(res)
cv2.imwrite(f"test_image_{res[0]}_{res[1]}.png", img)