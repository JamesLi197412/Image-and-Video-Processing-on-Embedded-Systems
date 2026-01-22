import numpy as np
import time
import cv2
import numpy as np
import time
from ex094 import grab_cut
from ex095 import add_alpha_channel

def paste_image(ag_img: np.ndarray, b_img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    if ag_img.ndim != 3 or ag_img.shape[2] != 4:
        raise ValueError("paste_image expects ag_img as BGRA (H, W, 4).")
    if b_img.ndim != 3 or b_img.shape[2] != 3:
        raise ValueError("paste_image expects b_img as BGR (H, W, 3).")

    roi_ag = ag_img[y:y+h, x:x+w]
    roi_b = b_img[y:y+h, x:x+w]

    alpha_ag = roi_ag[:, :, 3].astype(np.float32) / 255.0  # opacity of foreground
    alpha_b = 1.0 - alpha_ag

    for c in range(3):
        roi_b[:, :, c] = (alpha_ag * roi_ag[:, :, c].astype(np.float32) +
                          alpha_b * roi_b[:, :, c].astype(np.float32)).astype(np.uint8)

    b_img[y:y+h, x:x+w] = roi_b
    return b_img

if __name__ == '__main__':
    cap = cv2.VideoCapture("fg.mp4")
    bg_frame = cv2.VideoCapture("bg.mp4")

    object_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    ret, frame = cap.read()
    ret_bg, frame_bg = bg_frame.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_objects = object_classifier.detectMultiScale(gray, 1.1, 5)

    if len(detected_objects) == 0:
        print("No objects detected")
    else:
        for (x, y, w, h) in detected_objects:
            gc = grab_cut(frame, x, y, w, h, iter_count=5)
            ag = add_alpha_channel(gc)

            t0 = time.perf_counter()
            composite = paste_image(ag, bg_frame, x, y, w, h)
            t1 = time.perf_counter()
            print("paste_image runtime:", t1 - t0)

            #
            cv2.imwrite("alpha_bgra.png", ag)
            cv2.imwrite("composite.png", composite)

    cap.release()
    bg_frame.release()
    cv2.destroyAllWindows()