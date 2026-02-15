from __future__ import annotations

import argparse

from edge_ops import (
    load_gray_image,
    plot_filter_family,
    plot_gaussian_derivatives,
    plot_sigma_variation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise 5.4: Laplace/Gauss family comparisons.")
    parser.add_argument("--input", default="chessboard_crooked.png", help="Input grayscale image path.")
    parser.add_argument("--sigma", type=float, default=5.0, help="Sigma for Gaussian-based filters.")
    parser.add_argument("--output-main", default="laplace_filtered.png", help="Output for main filter grid.")
    parser.add_argument(
        "--output-smoothing",
        default="gaussian_smoothing_comparison.png",
        help="Output for smoothing sigma comparison.",
    )
    parser.add_argument(
        "--output-derivatives",
        default="gaussian_gradient_comparison.png",
        help="Output for gradient sigma comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        img = load_gray_image(args.input)
        plot_filter_family(img, args.output_main, sigma=args.sigma)
        sigma_values = [1, 3, 7, 10]
        plot_sigma_variation(img, args.output_smoothing, sigma_values=sigma_values)
        plot_gaussian_derivatives(img, args.output_derivatives, sigma_values=sigma_values)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Saved: {args.output_main}")
    print(f"Saved: {args.output_smoothing}")
    print(f"Saved: {args.output_derivatives}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
