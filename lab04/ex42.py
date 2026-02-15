from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image
from PIL.Image import Image as PILImage


# exercise 4.2
# The conclusion: pixel-by-pixel loops are simple but slower than vectorized approaches.
def convert_to_grayscale_simple(img: PILImage) -> PILImage:
    width, height = img.size
    gray_img = Image.new("L", (width, height))

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            luminance = (r + g + b) // 3
            gray_img.putpixel((x, y), luminance)
    return gray_img


def convert_to_grayscale_weighted(img: PILImage) -> PILImage:
    width, height = img.size
    gray_img = Image.new("L", (width, height))

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))
            luminance = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_img.putpixel((x, y), luminance)

    return gray_img


def _load_rgb_image(image_path: Path) -> PILImage:
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    img = Image.open(image_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        img = img.convert("RGB")
    return img


def process_image(image_path: Path, output_dir: Path) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    img = _load_rgb_image(image_path)

    gray_simple = convert_to_grayscale_simple(img)
    gray_weighted = convert_to_grayscale_weighted(img)

    stem = image_path.stem
    simple_path = output_dir / f"{stem}_simple.png"
    weighted_path = output_dir / f"{stem}_weighted.png"

    gray_simple.save(simple_path)
    gray_weighted.save(weighted_path)

    return simple_path, weighted_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an image to grayscale (simple + weighted).")
    parser.add_argument("--input", default="mandrill.png", help="Path to input image.")
    parser.add_argument("--output-dir", default=".", help="Directory for generated outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    try:
        simple_path, weighted_path = process_image(input_path, output_dir)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved simple grayscale image: {simple_path}")
    print(f"Saved weighted grayscale image: {weighted_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
