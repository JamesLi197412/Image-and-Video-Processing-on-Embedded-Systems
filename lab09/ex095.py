import cv2
import numpy as np
import time
from ex094 import grab_cut

def add_alpha_channel(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("add_alpha_channel expects a BGR image (H, W, 3).")

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    black = np.all(bgra[:, :, :3] == 0, axis=2)
    bgra[black, 3] = 0  # alpha channel

    return bgra

if __name__ == '__main__':
    cap = cv2.VideoCapture("fg.mp4")
    object_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_objects = object_classifier.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in detected_objects:
        gc = grab_cut(frame, x, y, w, h, iter_count=5)
        t0 = time.perf_counter()
        ag = add_alpha_channel(gc)
        t1 = time.perf_counter()
        print("add_alpha_channel runtime:", t1 - t0)
        cv2.imwrite("add_alpha_channel.png", ag)

    cap.release()
    cv2.destroyAllWindows()