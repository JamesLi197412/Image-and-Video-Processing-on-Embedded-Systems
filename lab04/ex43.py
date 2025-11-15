import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from typing import List

def load_image_to_array(file_name:str) -> np.ndarray:
    try:
        img = Image.open(file_name)
        img_array = np.array(img, dtype = np.uint8)
        return img_array
    except FileNotFoundError:
        print("File not found")

def save_array_to_gray_img(np_array_img:np.ndarray, file_name:str):
    assert isinstance(np_array_img, np.ndarray), "It has to be a Numpy array"
    assert isinstance(file_name, str), "It has to be a string"
    assert np_array_img.dtype == np.uint8, f"Array dtype must be uint8,  got {np_array_img.dtype}"

    img = Image.fromarray(np_array_img)
    img.save(file_name)

def convert_rgb_equal(img_rgb:np.ndarray) -> np.ndarray:
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    luminance = (R + G + B) / 3
    return luminance.astype(np.uint8)

def convert_rgb_weighted_1(img_rgb:np.ndarray) -> np.ndarray:
    R = img_rgb[:, :, 0]
    G = img_rgb[:, :, 1]
    B = img_rgb[:, :, 2]

    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    return luminance.astype(np.uint8)

def convert_rgb_weighted_2(img_rgb:np.ndarray) -> np.ndarray:
    weights = np.array([0.229, 0.587, 0.114])
    luminance = np.dot(img_rgb, weights)
    return luminance.astype(np.uint8)

# User - defined hist function
def hist_self(img_gray: np.ndarray, bins: int = 256) -> List[int]:
    hist = [0] * bins
    for pixel in img_gray.flatten():
        hist[pixel] += 1

    return hist

def plot_histogram(img_gray: np.ndarray, plot_path: str):
    """
    Plot histogram of a grayscale image and save it.

    Args:
        img_gray: Grayscale image as NumPy array
        plot_path: Path to save the histogram plot
    """
    assert isinstance(img_gray, np.ndarray), "img_gray must be a NumPy array"
    assert img_gray.dtype == np.uint8, f"Image must be uint8, got {img_gray.dtype}"
    assert len(img_gray.shape) == 2, f"Image must be grayscale (2D), got shape {img_gray.shape}"

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Manual histogram implementation
    pixel_values = img_gray.flatten()
    #ax1.hist(pixel_values, bins=256, range=(0, 255), color='gray', edgecolor='black', alpha=0.7)
    counts = hist_self(img_gray)
    ax1.plot(range(256), counts, color ='gray', linewidth=2)
    ax1.set_title('Histogramm 1')
    ax1.set_xlabel('Intensity Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Matplotlib's hist() function
    ax2.hist(pixel_values, bins=256, range=(0, 255), color='blue', edgecolor='black', alpha=0.7)
    ax2.set_title('Histogram 2')
    ax2.set_xlabel('Intensity Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved histogram to {plot_path}")
    plt.close()

def comparision(filename:str):
    mandrill = np.array(Image.open(filename), dtype = np.uint8)
    mandrill_gray = convert_rgb_weighted_2(mandrill)
    plot_histogram(mandrill_gray, f"Histogram for mandrill.png")



comparision("mandrill.png")