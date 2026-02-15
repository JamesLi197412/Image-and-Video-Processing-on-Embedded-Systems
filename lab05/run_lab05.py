from __future__ import annotations

import argparse
from pathlib import Path

from edge_ops import (
    load_gray_image,
    plot_filter_family,
    plot_gaussian_derivatives,
    plot_gradient_comparison,
    plot_sigma_variation,
    plot_sobel_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run complete Lab05 edge-detection comparison pipeline.")
    parser.add_argument("--input-basic", default="chessboard.pbm", help="Input image for gradient comparison.")
    parser.add_argument(
        "--input-crooked",
        default="chessboard_crooked.png",
        help="Input image for Sobel/Laplace/LoG comparisons.",
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for plots.")
    parser.add_argument("--sigma", type=float, default=5.0, help="Sigma for Gaussian-based filters.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_basic = load_gray_image(args.input_basic)
        img_crooked = load_gray_image(args.input_crooked)

        plot_gradient_comparison(img_basic, str(out_dir / "filtered.png"))
        plot_gradient_comparison(img_crooked, str(out_dir / "filtered_crooked.png"))
        plot_sobel_comparison(img_crooked, str(out_dir / "sobel_filtered.png"))
        plot_filter_family(img_crooked, str(out_dir / "laplace_filtered.png"), sigma=args.sigma)
        plot_sigma_variation(
            img_crooked,
            str(out_dir / "gaussian_smoothing_comparison.png"),
            sigma_values=[1, 3, 7, 10],
        )
        plot_gaussian_derivatives(
            img_crooked,
            str(out_dir / "gaussian_gradient_comparison.png"),
            sigma_values=[1, 3, 7, 10],
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Lab05 outputs saved to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
