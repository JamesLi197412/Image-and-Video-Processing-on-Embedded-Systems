import numpy as np
import cv2
import time

def grab_cut(frame: np.ndarray, x: int, y: int, w: int, h: int, iter_count: int = 5) -> np.ndarray:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    rect = (x, y, w, h)

    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, iterCount=iter_count, mode=cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype("uint8")
    result = frame * mask2[:, :, np.newaxis]
    return result

if __name__ == "__main__":
    cap = cv2.VideoCapture("fg.mp4")
    object_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detected_objects = object_classifier.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in detected_objects:
        t0 = time.perf_counter()
        gc = grab_cut(frame, x, y, w, h, iter_count=5)
        t1 = time.perf_counter()
        print("grab_cut runtime:", t1 - t0)

    cap.release()
