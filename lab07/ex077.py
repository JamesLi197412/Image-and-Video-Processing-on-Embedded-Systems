import cv2
import numpy as np
import psutil
import threading
import time
import sys

from ex072 import interface
from ex076 import create_panorama
DEBUG = True

def show_free_memory(stop: threading.Event):
    " Present free memory during execution"
    while not stop.is_set():
        mem = psutil.virtual_memory()
        print(f"Free memory: {mem.available / (1024**3):.2f} GB")
        time.sleep(0.5)

if DEBUG:
    try:
        stop_event = threading.Event()
        thread_id = threading.Thread(target=show_free_memory, args=(stop_event,))
        print("DEBUG MODE ON")
    except Exception:
        print("thread creation failed")
        sys.exit(0)

imgs = interface()
assert len(imgs) > 1

_,_,_, result = create_panorama(imgs[0], imgs[1])
for next_img in imgs[2:]:
    result = create_panorama(result, next_img)

cv2.imwrite("panorama.png", result)

if DEBUG:
    stop_event.set()
    thread_id.join()