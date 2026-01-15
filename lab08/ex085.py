import cv2
from gpiozero import Button
from picamera2 import Picamera2
import select
import sys
import time
import numpy as np

red_button = Button(6)
pencasc_vert = cv2.CascadeClassifier("pen_vertical_classifier.xml")

T_LOW = 50
T_HIGH = 150

def hough_transform(x: int, y: int, w: int, h: int, image: np.ndarray):
    height, width = image.shape[:2]

    x1 = max(0, x - 200)
    y1 = max(0, y - 40)
    x2 = min(width, x + w + 200)
    y2 = min(height, y + h + 40)

    if x2 <= x1 and y2 <= y1:
        return None

    cropped_image = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray, T_LOW, T_HIGH, apertureSize=3)

    lines = cv2.HoughLines(edge_image, 1, np.pi / 180, 120)  #     Hough Transform


    if lines is not None:
        print("Line found!")
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho

            x1l = int(x0 + 1000 * (-b))
            y1l = int(y0 + 1000 * (a))
            x2l = int(x0 - 1000 * (-b))
            y2l = int(y0 - 1000 * (a))

            cv2.line(
                cropped_image,
                (x1l, y1l),
                (x2l, y2l),
                (0, 0, 255),
                2
            )

            if theta > np.pi / 2:
                angle = (theta - np.pi / 2) * 180 / np.pi
            else:
                angle = theta * 180 / np.pi

            cv2.putText(
                image,
                f"Angle: {angle:.2f}",
                (10, 25),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2
            )


with Picamera2() as picam2:
    config = picam2.create_preview_configuration(
        main={"format": "XRGB8888", "size": (640, 480)}
    )
    picam2.configure(config)
    picam2.start()

    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    time.sleep(1)

    while True:
        bgr_frame = picam2.capture_array()
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

        pens_vert = pencasc_vert.detectMultiScale(
            gray,
            scaleFactor=1.7,
            minNeighbors=25,
            minSize=(25, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for x, y, w, h in pens_vert:
            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            hough_transform(x, y, w, h, bgr_frame)

        cv2.imshow("window", bgr_frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
            break

        if select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.readline().strip().lower() == "q":
                break

        time.sleep(1 / 10)

    picam2.stop()

cv2.destroyAllWindows()
