import cv2
import select
import sys
import time
from gpiozero import Button
from picamera2 import Picamera2

# reference: https://docs.opencv.org/3.4.3/db/d5c/tutorial_py_bg_subtraction.html
def main():
    cap = cv2.VideoCapture(0)

    # Try MOG2 and KNN (OpenCV provides both)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400.0, detectShadows=True)

    use_knn = False

    while True:
        ret, frame = cap.read()

        subtractor = knn if use_knn else mog2
        fgmask = subtractor.apply(frame)

        cv2.imshow("Frame", frame)
        cv2.imshow("FG Mask", fgmask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('t'):
            use_knn = not use_knn

    cap.release()
    cv2.destroyAllWindows()

def show_video():
    """
        Background Subtraction method comparison
    :return:
    """
    red_button = Button(6)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # fgbg = cv2.createBackgroundSubtractorKNN()

    video_capture = None

    with Picamera2() as picam2:
        config = picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )

        if video_capture is None:
            picam2.configure(config)
            picam2.start()

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("FG Mask", cv2.WINDOW_NORMAL)

        time.sleep(1)

        while True:
            if video_capture is None:
                bgr_frame = picam2.capture_array()
            else:
                ret, bgr_frame = video_capture.read()
                if not ret:
                    break

            fgmask = fgbg.apply(bgr_frame)

            cv2.imshow("Original", bgr_frame)
            cv2.imshow("FG Mask", fgmask)

            if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
                break

            if select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == "q":
                    break

            if video_capture is not None:
                time.sleep(1 / 30)

        if video_capture is None:
            picam2.stop()
        else:
            video_capture.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    show_video()
