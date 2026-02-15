from typing import Tuple

from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage
import time

# Global variables
BLACK = (0,0,0)
WHITE = (255,255,255)
CHESS_SIZE = [8,8]
FIELD_SIZE = [5,5]
WIDTH = CHESS_SIZE[0] * FIELD_SIZE[0]
HEIGHT = CHESS_SIZE[1] * FIELD_SIZE[1]


def save_image(pic: PILImage, pic_name: str, pic_type:str) -> bool:
    assert isinstance(pic_name, str) and len(pic_name) > 0
    assert isinstance(pic_type, str) and len(pic_type) > 0

    try:
        pic.save(f"{pic_name}.{pic_type}")
        return True
    except:
        return False

def create_blank_image(pic_width: int, pic_height: int) -> PILImage:
    assert isinstance(pic_width, int) and isinstance(pic_height, int)
    assert(pic_width >0) and (pic_height >0)
    pic = Image.new("RGB", (pic_width, pic_height))
    return pic

def create_chessboard_1() -> PILImage:
    bw = [WHITE, BLACK]
    chess_field = create_blank_image(WIDTH, HEIGHT)
    #chess_field.show()
    for width in range(WIDTH):
        for height in range(HEIGHT):
            color_index = (width + height) % 2
            chess_field.putpixel((width, height), bw[color_index])

    return chess_field

def open_pic(file_path: str) -> PILImage:
    assert isinstance(file_path, str) and len(file_path) > 0

    pic = Image.open(file_path)
    return pic

def create_black_frame(pic: PILImage, frame_width: int) -> PILImage:
    assert isinstance(frame_width, int) and frame_width > 0

    orig_width, orig_height = pic.size

    # Create a black frame around the forwarded image pic
    new_width = orig_width + 2 * frame_width
    new_height = orig_height + 2 * frame_width
    pic_with_frame = create_blank_image(new_width, new_height)

    # Assign black
    for width in range(orig_width):
        for height in range(orig_height):
            color = pic.getpixel((width, height))
            pic_with_frame.putpixel((width + frame_width, height + frame_width), color)

    return pic_with_frame

def transpose_pic(pic: PILImage) -> PILImage:
    width, height = pic.size

    rotated_pic = create_blank_image(height, width)
    for x in range(width):
        for y in range(height):
            color = pic.getpixel((x, y))
            rotated_pic.putpixel((y, x), color)
    return rotated_pic


# Exercise 3.6
def create_bbox(position: Tuple[int, int]) -> Tuple[int, int, int, int]:
    x0 = position[0] * FIELD_SIZE[0]
    y0 = position[1] * FIELD_SIZE[1]
    x1 = x0 + FIELD_SIZE[0] - 1
    y1 = y0 + FIELD_SIZE[1] - 1

    # To make sure you do not exceed size
    x1 = min(x1, WIDTH - 1)
    y1 = min(y1, HEIGHT - 1)

    bbox = (x0, y0, x1, y1)
    return bbox

def create_chessboard_2()  -> PILImage:
    chessboard = create_blank_image(WIDTH, HEIGHT)
    draw = ImageDraw.Draw(chessboard)

    # Iterate over each chess square position
    for row in range(CHESS_SIZE[1]):
        for col in range(CHESS_SIZE[0]):
            # Determine if this square should be black (checkerboard pattern)
            if (row + col) % 2 == 1:
                # Get bounding box for this position
                bbox = create_bbox((col, row))
                # Fill the rectangle with black
                draw.rectangle(bbox, fill=BLACK)
    return chessboard



def create_chessboard_3() -> PILImage:
    chessboard = create_blank_image(WIDTH, HEIGHT)
    draw = ImageDraw.Draw(chessboard)

    for x in range(WIDTH):
        for y in range(HEIGHT):
            if (x + y) % 2 == 1:
                draw.point((x, y), fill=BLACK)
            else:
                draw.point((x, y), fill=WHITE)

    save_image(chessboard, "chessboard_3", "png")
    return chessboard


def check_time(time_in_sec: float) -> float:
    start = time.time()
    time.sleep(time_in_sec)
    stop = time.time()
    return stop - start

def measure_efficiency():
    print("Measuring create_chessboard_2()")
    start_2 = time.perf_counter()
    create_chessboard_2()
    elapsed_time_2 = time.perf_counter() - start_2
    print(f"create_chessboard_2 took: {elapsed_time_2:.6f} seconds")

    # Measure the time for create_chessboard_3 (using point method)
    print("\nMeasuring create_chessboard_3...")
    start_3 = time.perf_counter()
    create_chessboard_3()
    elapsed_time_3 = time.perf_counter() - start_3
    print(f"create_chessboard_3 took: {elapsed_time_3:.6f} seconds")

    # Compare the results
    print(f"\nSpeed comparison:")
    print(f"create_chessboard_2: {elapsed_time_2:.6f} seconds")
    print(f"create_chessboard_3: {elapsed_time_3:.6f} seconds")
