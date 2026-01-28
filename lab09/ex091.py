import cv2
# reference: https://docs.opencv.org/3.0.0/d7/d8b/tutorial_py_face_detection.html

import cv2
from gpiozero import Button
from picamera2 import Picamera2
import select
import sys
import time

def main(classifier, scale_factor, min_neighbors, min_size):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(classifier)
    if face_cascade.empty():
        raise RuntimeError(f"Could not load cascade")

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Haar Face Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def video_record(classifier, scale_factor, min_neighbors, min_size):
    red_button = Button(6)
    face_cascade = cv2.CascadeClassifier(classifier)

    video_capture = None

    with Picamera2() as picam2:
        config = picam2.create_preview_configuration(
            main={"format": "XRGB8888", "size": (640, 480)}
        )

        if video_capture is None:
            picam2.configure(config)
            picam2.start()

        cv2.namedWindow("window", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        time.sleep(1)

        while True:
            if video_capture is None:
                bgr_frame = picam2.capture_array()
            else:
                ret, bgr_frame = video_capture.read()
                if not ret:
                    break

            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

            # TODO: 2. 检测人脸
            # 练习 9.1 要求调整参数以获得最佳结果
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=min_size,
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # TODO: 3. 在检测到的物体周围画矩形
            # 这里不需要 itertools 了，因为我们只用了一个分类器
            for (x, y, w, h) in faces:
                # 在原图上画矩形，颜色为绿色 (0, 255, 0)，线宽为 2
                cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("window", bgr_frame)

            if cv2.waitKey(1) & 0xFF == ord("q") or red_button.is_pressed:
                break

            if select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == "q":
                    break

            if video_capture is not None:
                time.sleep(1 / 30)  # 模拟 30fps

        if video_capture is None:
            picam2.stop()
        else:
            video_capture.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    classifier = 'haarcascade_frontalface_default.xml'
    scale_factor = 1.1
    min_neighbors = 5
    min_size = (30,30)
    #main(classifier, scale_factor, min_neighbors,min_size )
    video_record(classifier,scale_factor,min_neighbors,min_size)
