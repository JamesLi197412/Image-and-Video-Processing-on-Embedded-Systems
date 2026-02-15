from __future__ import annotations

import argparse

from edge_ops import load_gray_image, plot_sobel_comparison, sobel_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 5.3: Sobel operator comparisons.")
    parser.add_argument("--input", default="chessboard_crooked.png", help="Input grayscale image path.")
    parser.add_argument("--output", default="sobel_filtered.png", help="Output figure path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        img = load_gray_image(args.input)
        plot_sobel_comparison(img, args.output)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
