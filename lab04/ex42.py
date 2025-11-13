from PIL import Image
from PIL.Image import Image as PILImage
from typing import List

#exercise 4.2
# The conclusion: You are strongly not suggested to work on pixel by pixel, it will take too longer time.
def convert_to_grayscale_simple(img: PILImage) -> PILImage:
    width, height = img.size
    gray_img = Image.new('L', (width, height))

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            luminance = (r + g + b) // 3
            gray_img.putpixel((x, y), luminance)
    return gray_img


def convert_to_grayscale_weighted(img: PILImage) -> PILImage:
    width, height = img.size
    gray_img = Image.new('L', (width, height))

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            luminance = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_img.putpixel((x, y), luminance)

    return gray_img


def histogram(img_gray: PILImage) -> List[int]:
    width, height = img_gray.size
    pixel_values = []

    for y in range(height):
        for x in range(width):
            pixel_values.append(img_gray.getpixel((x, y)))

    return pixel_values

def image_process(filename:str):
    img = Image.open( filename +".png")
    gray_simple = convert_to_grayscale_simple(img)
    gray_weighted = convert_to_grayscale_weighted(img)

    gray_simple.save(f"{filename}_simple.png")
    gray_weighted.save(f"{filename}_weighted.png")

image_process("mandrill")