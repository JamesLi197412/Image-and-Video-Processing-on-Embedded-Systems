# Exercise 8.1 (Raspberry Pi)
import cv2
from gpiozero import Button
import select
import sys

# rpicam-vid --framerate 30 -t 0 --width 640 --height 480 --inline --codec libav --libav-format avi -o -my_video.avi

VIDEO_SOURCE = "my_video.avi"
video_capture = cv2.VideoCapture(VIDEO_SOURCE)

red_button = Button(6)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    cv2.imshow("video", gray)

    if (cv2.waitKey(1) & 0xFF) == ord("q") or red_button.is_pressed:
        break

    if select.select([sys.stdin], [], [], 0)[0]:
        if sys.stdin.readline().strip().lower() == "q":
            break

video_capture.release()
cv2.destroyAllWindows()