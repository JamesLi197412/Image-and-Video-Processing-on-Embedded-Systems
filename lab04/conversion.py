from PIL import Image
from PIL.Image import Image as PILImage


#exercise 4.2
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

img_0 = Image.open("image_0.png")
gray_0_simple = convert_to_grayscale_simple(img_0)
gray_0_weighted = convert_to_grayscale_weighted(img_0)

gray_0_simple.save("gray_0_simple.png")
gray_0_weighted.save("gray_0_weighted.png")