import cv2
from grabcut import grab_cut
from alphachannel import add_alpha_channel
from pasteImage import paste_image
import time

def background_replacement(input_video_path: str, background_video_path: str, weightsfile: str) -> None:
    cap = cv2.VideoCapture(input_video_path)
    bg_cap = cv2.VideoCapture(background_video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Video Does not exist: {input_video_path}")
    if not bg_cap.isOpened():
        raise RuntimeError(f"Background Video Does not Exist: {background_video_path}")

    object_classifier = cv2.CascadeClassifier(weightsfile)
    if object_classifier.empty():
        raise RuntimeError(f"Could not load cascade file: {weightsfile}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 10.0


    frame_counter = 0
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter("out.mp4", fourcc, fps, (width, height))

    fg_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bg_count = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(fg_count, bg_count)

    while frame_counter < max_frames:
        ret, frame = cap.read()
        success, background_frame = bg_cap.read()

        if not ret or not success:
            break

        if background_frame.shape[:2] != (height, width):
            background_frame = cv2.resize(background_frame, (width, height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected_objects = object_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for x, y, w, h in detected_objects:
            gc = grab_cut(frame, x, y, w, h, iter_count=5)
            cropped = gc[y:y+h, x:x+w]  # to save computation
            ag = add_alpha_channel(cropped)
            background_frame = paste_image(ag, background_frame, x, y, w, h)
            break

        out.write(background_frame)
        frame_counter += 1

    cap.release()
    bg_cap.release()
    out.release()






if __name__ == "__main__":
    t0 = time.perf_counter()
    background_replacement("fg.mp4", "bg.mp4", "haarcascade_frontalface_default.xml")
    t1 = time.perf_counter()
    print(f"Total runtime: {t1 - t0:.6f} s")

# ffmpeg -y -i video.mp4 -r 10 -s 640x360 -c:v libx264 -b:v 1M -strict -2 -movflags faststart output_video.mp4