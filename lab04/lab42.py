from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def channel_visualization(input_path: str, output_path: str) -> None:
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    mandrill_rgb = np.array(Image.open(in_path), dtype=np.uint8)

    str_img_size = str(mandrill_rgb.size)
    str_img_height = str(mandrill_rgb.shape[0])
    str_img_width = str(mandrill_rgb.shape[1])
    str_img_channels = str(mandrill_rgb.shape[2])
    str_img_datatype = str(mandrill_rgb.dtype)

    mandrill_red_ch = mandrill_rgb.copy()
    mandrill_green_ch = mandrill_rgb.copy()
    mandrill_blue_ch = mandrill_rgb.copy()

    mandrill_red_ch[:, :, 1:3] = 0
    mandrill_green_ch[:, :, 0] = 0
    mandrill_green_ch[:, :, 2] = 0
    mandrill_blue_ch[:, :, :2] = 0

    plt.subplot(2, 3, 1)
    plt.imshow(mandrill_rgb)
    plt.title("Mandrill RGB")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.text(0, 0.8, "Image Size: ")
    plt.text(0, 0.6, "Image Width: ")
    plt.text(0, 0.5, "Image Height: ")
    plt.text(0, 0.3, "Image Channels: ")
    plt.text(0, 0.2, "Image Datatype: ")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.text(0, 0.8, str_img_size)
    plt.text(0, 0.6, str_img_width)
    plt.text(0, 0.5, str_img_height)
    plt.text(0, 0.3, str_img_channels)
    plt.text(0, 0.2, str_img_datatype)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(mandrill_red_ch)
    plt.title("Red Channel")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(mandrill_green_ch)
    plt.title("Green Channel")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(mandrill_blue_ch)
    plt.title("Blue Channel")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RGB channels and image metadata.")
    parser.add_argument("--input", default="mandrill.png", help="Input image path.")
    parser.add_argument("--output", default="mandrill_channels.png", help="Output plot path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        channel_visualization(args.input, args.output)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Saved channel visualization: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
