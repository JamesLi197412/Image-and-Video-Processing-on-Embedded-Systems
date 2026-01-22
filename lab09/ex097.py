import cv2
import numpy as np
from ex094 import grab_cut
from ex095 import add_alpha_channel
from ex096 import paste_image
import time

def background_replacement(input_video_path: str, background_video_path: str, weightsfile: str) -> None:
    cap = cv2.VideoCapture(input_video_path)
    background_cap = cv2.VideoCapture(background_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video_path}")
    if not background_cap.isOpened():
        raise RuntimeError(f"Could not open background video: {background_video_path}")

    object_classifier = cv2.CascadeClassifier(weightsfile)
    if object_classifier.empty():
        raise RuntimeError(f"Could not load cascade file: {weightsfile}")

    frame_counter = 0
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prefer using source FPS if available; fallback to 10.0 (as in the sheet)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0

    out = cv2.VideoWriter("out.mp4", fourcc, fps, (width, height))

    fg_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 10**9
    bg_count = int(background_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if background_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 10**9
    max_frames = min(fg_count, bg_count)

    while frame_counter < max_frames:
        ret, frame = cap.read()
        success, background_frame = background_cap.read()

        if not ret or not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_objects = object_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for x, y, w, h in detected_objects:
            gc = grab_cut(frame, x, y, w, h, iter_count=5)
            ag = add_alpha_channel(gc)
            background_frame = paste_image(ag, background_frame, x, y, w, h)

        out.write(background_frame)
        frame_counter += 1

    cap.release()
    background_cap.release()
    out.release()



if __name__ == "__main__":
    t0 = time.perf_counter()
    background_replacement("fg.mp4", "bg.mp4", "haarcascade_frontalface_default.xml")
    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.6f} s")

# ffmpeg -y -i video.mp4 -r 10 -s 640x360 -c:v libx264 -b:v 1M -strict -2 -movflags faststart output_video.mp4