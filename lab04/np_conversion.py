# using the np.array to convert or adjust the img

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def convert_to_grayscale(img):
    img_array = np.array(img)
    R,G,B = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    luminance = (R+G+B)/3
    return Image.fromarray(luminance.astype(np.uint8))

def convert_to_graysacle_weight(img):
    img_array = np.array(img)
    R,G,B = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    return Image.fromarray(luminance.astype(np.uint8))

def histogram(img_gray):
    img_array = np.array(img_gray)
    pixel_values = img_array.flatten().tolist()
    return pixel_values
