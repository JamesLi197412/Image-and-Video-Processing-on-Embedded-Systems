import cv2
from gpiozero import Button
from picamera2 import Picamera2
import select
import sys
import time

red_button = Button(6)

num_frames = 200
cnt = 0

RESOLUTION = (640, 480)
# RESOLUTION = (1280,960)

with Picamera2() as picam2:
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": RESOLUTION}
    )
    picam2.configure(config)
    picam2.start()

    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    time.sleep(1)

    start = time.time()

    # --- TASK: change processing FPS here (simulated) ---
    SLEEP_SEC = 1 / 10  # try 0 (fastest), 1/30, 1/5, etc.

    while True:
        cnt += 1

        # (2) print FPS every num_frames frames
        if cnt == num_frames:
            end = time.time()
            seconds = end - start
            fps = num_frames / seconds if seconds > 0 else 0
            print(f"Time taken = {seconds:.3f} seconds.")
            print(f"Estimated FPS = {fps:.2f} fps.")
            cnt = 0
            start = time.time()

        bgr_frame = picam2.capture_array()  # numpy array
        cv2.imshow("window", bgr_frame)

        # exit conditions
        if (cv2.waitKey(1) & 0xFF) == ord("q") or red_button.is_pressed:
            break

        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == "q":
                break

        time.sleep(SLEEP_SEC)

    picam2.stop()

cv2.destroyAllWindows()
