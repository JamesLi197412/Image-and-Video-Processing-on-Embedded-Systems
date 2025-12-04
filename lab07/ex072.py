import cv2
import select
import sys
import time
from gpiozero import Button
from picamera2 import Picamera2
import numpy as np
from typing import List, Tuple, Optional

INFO = {
    "start": "Starting panorama program.",
    "init": "Initializing camera and buttons...",
    "init_ok": "Initialization succeeded.",
    "init_fail": "A critical error occurred. Restarting session...",
    "settings_ok": "Settings confirmed. Starting capture session.",
    "cam_error": "A camera error occurred. Restarting session...",
    "exit_ok": "Camera closed successfully!",
    "take_img": "Press the GREEN button to take a picture. Press RED or 'q' to exit.",
    "res_fail": "Invalid resolution format. Please use 'width,height' (e.g., 1920,1080).",
    "frame_count_fail": "Invalid number. Defaulting to 4 frames.",
}

USER_INPUT = {
    "res": "Enter resolution as 'width,height': ",
    "frame_count": "How many frames to capture (max 6)? ",
}

def get_user_settings() -> Tuple[Tuple[int, int], int]:
    frame_number = 0
    resolution = 0
    while True:
        try:
            res_input = input(USER_INPUT["res"])
            width_str, height_str = res_input.split(',')
            resolution: Tuple[int, int] = (int(width_str), int(height_str))
            break
        except ValueError:
            print(INFO["res_fail"])

    while True:
        try:
            frame_number = int(input(USER_INPUT["frame_count"]))
            if not 0 < frame_number <= 6:
                raise ValueError
            break
        except (ValueError, TypeError):
            frame_number = 4
            print(INFO["frame_count_fail"])

    print(INFO["settings_ok"])
    return resolution, frame_number

def capture_sequence(resolution: Tuple[int, int],frame_count: int) -> Optional[List[np.ndarray]]:
    print(INFO["init"])
    frames = []

    # initialisation
    try:
        with Picamera2() as picam2:
            config = picam2.create_preview_configuration(
                main={"format": "XRGB8888", "size": resolution}
            )
            picam2.configure(config)

            green_button = Button(5)
            red_button = Button(6)

            picam2.start()
            print(INFO["init_ok"])

            cv2.namedWindow("panorama", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                "panorama", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )

            time.sleep(1)
    except Exception as e:
        print(f"\nError: {e}")
        return None

    is_pressed = False
    frames_remaining = frame_count

    try:
        while frames_remaining > 0:
            bg_frame = picam2.capture_array()
            cv2.imshow("panorama", bg_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Capture cancelled by user.")
                return []

            # If user presses q in terminal
            if select.select([sys.stdin], [], [], 0)[0]:
                if sys.stdin.readline().strip().lower() == "q":
                    print("Capture cancelled by user.")
                    return []

            # RED button → exit
            if red_button.is_pressed() and not is_pressed:
                print("Capture cancelled by user.")
                return []

            # GREEN button → take picture (only on state change!)
            if green_button.is_pressed() and not is_pressed:
                frames.append(bg_frame[:, :, :3])
                frames_remaining -= 1
                print(f"Picture taken! ({frames_remaining} remaining.)")

            # update last state of buttons
            is_pressed = green_button.is_pressed() or red_button.is_pressed()

            return frames

    except Exception as e:
        print(f"{INFO['cam_err']} {e}")
        return None

    finally:
        cv2.destroyAllWindows()
        print(INFO["exit_ok"])


def interface():
    print(INFO["start"])
    print(INFO["init"])

    resolution, frame_number = get_user_settings()
    while True:

        captured_frames = capture_sequence(resolution, frame_number)

        if captured_frames is None:
            time.sleep(2)
            continue

        if captured_frames:
            print(f"\nSuccessfully captured {len(captured_frames)}Exiting.")
        else:
            print(f"\nSuccessfully captured cancelled by user. Exiting.")
        break

if __name__ == '__main__':
    interface()