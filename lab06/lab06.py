from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Union

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

mpl.use("agg")


def create_test_img(width: int = 800, height: int = 800) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.float32)
    img[:] = 255
    img[0:400, 0:400] = 0
    return img


def add_noise(img: np.ndarray, dev: Union[int, float]) -> np.ndarray:
    noise = np.random.normal(loc=0, scale=dev, size=img.shape)
    n_img = np.clip(img + noise, 0, 255)
    return n_img.astype(np.float32)


def compute_derivatives(img: np.ndarray, sigma: Union[int, float] = 3) -> Tuple[np.ndarray, np.ndarray]:
    fm = gaussian_filter(img, sigma=(sigma, sigma), order=(0, 1))
    fn = gaussian_filter(img, sigma=(sigma, sigma), order=(1, 0))
    return fm, fn


def window_img(
    img_window: Tuple[int, int, int, int],
    img: np.ndarray,
    fm: np.ndarray,
    fn: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_size = 10
    red = np.array([255, 0, 0], dtype=np.uint8)
    m0, n0, m1, n1 = img_window

    img_with_frame = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img_with_frame[m0 : m0 + frame_size, n0:n1] = red
    img_with_frame[m1 - frame_size : m1, n0:n1] = red
    img_with_frame[m0:m1, n0 : n0 + frame_size] = red
    img_with_frame[m0:m1, n1 - frame_size : n1] = red

    w_fm = fm[m0:m1, n0:n1]
    w_fn = fn[m0:m1, n0:n1]
    return w_fm, w_fn, img_with_frame


def ex61(dev: float, sigma: float):
    img = create_test_img(800, 800)
    n_img = add_noise(img, dev)

    fm, fn = compute_derivatives(img, sigma)
    fm_n, fn_n = compute_derivatives(n_img, sigma)

    flat = (600, 200, 700, 300)
    edge = (350, 100, 450, 200)
    corner = (350, 350, 450, 450)

    img_uint8 = img.astype(np.uint8)
    w_fm1, w_fn1, img1 = window_img(flat, img_uint8, fm, fn)
    w_fm2, w_fn2, img2 = window_img(edge, img_uint8, fm, fn)
    w_fm3, w_fn3, img3 = window_img(corner, img_uint8, fm, fn)

    w_fm4, w_fn4, img4 = window_img(flat, img_uint8, fm_n, fn_n)
    w_fm5, w_fn5, img5 = window_img(edge, img_uint8, fm_n, fn_n)
    w_fm6, w_fn6, img6 = window_img(corner, img_uint8, fm_n, fn_n)

    return (
        w_fm1,
        w_fn1,
        img1,
        w_fm2,
        w_fn2,
        img2,
        w_fm3,
        w_fn3,
        img3,
        w_fm4,
        w_fn4,
        img4,
        w_fm5,
        w_fn5,
        img5,
        w_fm6,
        w_fn6,
        img6,
        img,
        n_img,
    )


def exercise061_plot(*args) -> None:
    (
        w_fm1,
        w_fn1,
        img1,
        w_fm2,
        w_fn2,
        img2,
        w_fm3,
        w_fn3,
        img3,
        w_fm4,
        w_fn4,
        img4,
        w_fm5,
        w_fn5,
        img5,
        w_fm6,
        w_fn6,
        img6,
        img,
        n_img,
    ) = args

    plt.figure(figsize=(18, 12))

    plt.subplot(4, 6, 1), plt.title("noiseless"), plt.axis("off"), plt.imshow(img, cmap="gray")
    plt.subplot(4, 6, 4), plt.title("noisy"), plt.axis("off"), plt.imshow(n_img, cmap="gray")

    plt.subplot(4, 6, 7), plt.title("flat"), plt.axis("off"), plt.imshow(img1)
    plt.subplot(4, 6, 8), plt.title("Fm"), plt.axis("off"), plt.imshow(w_fm1, cmap="gray")
    plt.subplot(4, 6, 9), plt.title("Fn"), plt.axis("off"), plt.imshow(w_fn1, cmap="gray")
    plt.subplot(4, 6, 10), plt.title("noisy flat"), plt.axis("off"), plt.imshow(img4)
    plt.subplot(4, 6, 11), plt.title("Fm"), plt.axis("off"), plt.imshow(w_fm4, cmap="gray")
    plt.subplot(4, 6, 12), plt.title("Fn"), plt.axis("off"), plt.imshow(w_fn4, cmap="gray")

    plt.subplot(4, 6, 13), plt.title("edge"), plt.axis("off"), plt.imshow(img2)
    plt.subplot(4, 6, 14), plt.title("Fm"), plt.axis("off"), plt.imshow(w_fm2, cmap="gray")
    plt.subplot(4, 6, 15), plt.title("Fn"), plt.axis("off"), plt.imshow(w_fn2, cmap="gray")
    plt.subplot(4, 6, 16), plt.title("noisy edge"), plt.axis("off"), plt.imshow(img5)
    plt.subplot(4, 6, 17), plt.title("Fm"), plt.axis("off"), plt.imshow(w_fm5, cmap="gray")
    plt.subplot(4, 6, 18), plt.title("Fn"), plt.axis("off"), plt.imshow(w_fn5, cmap="gray")

    plt.subplot(4, 6, 19), plt.title("corner"), plt.axis("off"), plt.imshow(img3)
    plt.subplot(4, 6, 20), plt.title("Fm"), plt.axis("off"), plt.imshow(w_fm3, cmap="gray")
    plt.subplot(4, 6, 21), plt.title("Fn"), plt.axis("off"), plt.imshow(w_fn3, cmap="gray")
    plt.subplot(4, 6, 22), plt.title("noisy corner"), plt.axis("off"), plt.imshow(img6)
    plt.subplot(4, 6, 23), plt.title("Fm"), plt.axis("off"), plt.imshow(w_fm6, cmap="gray")
    plt.subplot(4, 6, 24), plt.title("Fn"), plt.axis("off"), plt.imshow(w_fn6, cmap="gray")

    plt.tight_layout()
    plt.savefig("Noiseless_Noisy.png", dpi=150)
    plt.close()


def exercise062_plot(*args) -> None:
    (
        w_fm1,
        w_fn1,
        _img1,
        w_fm2,
        w_fn2,
        _img2,
        w_fm3,
        w_fn3,
        _img3,
        w_fm4,
        w_fn4,
        _img4,
        w_fm5,
        w_fn5,
        _img5,
        w_fm6,
        w_fn6,
        _img6,
        _img,
        _n_img,
    ) = args

    plt.figure(figsize=(18, 12))
    plt.subplot(321), plt.title("flat"), plt.ylabel("Fn"), plt.axis([-10, 10, -10, 10]), plt.grid(True)
    plt.plot(w_fm1.flatten(), w_fn1.flatten(), "rx")

    plt.subplot(322), plt.title("noisy flat"), plt.axis([-10, 10, -10, 10]), plt.grid(True)
    plt.plot(w_fm4.flatten(), w_fn4.flatten(), "rx")

    plt.subplot(323), plt.title("edge"), plt.ylabel("Fn"), plt.axis([-40, 40, -40, 40]), plt.grid(True)
    plt.plot(w_fm2.flatten(), w_fn2.flatten(), "rx")

    plt.subplot(324), plt.title("noisy edge"), plt.axis([-40, 40, -40, 40]), plt.grid(True)
    plt.plot(w_fm5.flatten(), w_fn5.flatten(), "rx")

    plt.subplot(325), plt.title("corner"), plt.ylabel("Fn"), plt.xlabel("Fm")
    plt.axis([-40, 40, -40, 40]), plt.grid(True)
    plt.plot(w_fm3.flatten(), w_fn3.flatten(), "rx")

    plt.subplot(326), plt.title("noisy corner"), plt.xlabel("Fm")
    plt.axis([-40, 40, -40, 40]), plt.grid(True)
    plt.plot(w_fm6.flatten(), w_fn6.flatten(), "rx")

    plt.tight_layout()
    plt.savefig("eigen.png")
    plt.close()


def compute_harris_corners(
    fm: np.ndarray,
    fn: np.ndarray,
    k: Union[int, float] = 0.04,
    sigma: Union[int, float] = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mmm = gaussian_filter(fm * fm, sigma)
    mmn = gaussian_filter(fm * fn, sigma)
    mnm = mmn
    mnn = gaussian_filter(fn * fn, sigma)

    det_m = mmm * mnn - mmn * mnm
    tra_m = mmm + mnn
    r = det_m - k * (tra_m ** 2)

    return mmm, mmn, mnm, mnn, det_m, tra_m, r


def figure067(
    img: np.ndarray,
    n_img: np.ndarray,
    det_m: np.ndarray,
    tra_m: np.ndarray,
    r: np.ndarray,
    det_m_n: np.ndarray,
    tra_m_n: np.ndarray,
    r_n: np.ndarray,
) -> None:
    ax1 = plt.subplot(2, 4, 1)
    ax1.set_title("image")
    ax1.axis("off")
    ax1.imshow(img, cmap="gray")

    ax2 = plt.subplot(2, 4, 2)
    ax2.set_title("corner response R")
    ax2.axis("off")
    ax2.imshow(r, cmap="gray")

    ax3 = plt.subplot(2, 4, 5)
    ax3.set_title("det(M)")
    ax3.axis("off")
    ax3.imshow(det_m, cmap="gray")

    ax4 = plt.subplot(2, 4, 6)
    ax4.set_title("trace(M)")
    ax4.axis("off")
    ax4.imshow(tra_m, cmap="gray")

    ax5 = plt.subplot(2, 4, 3)
    ax5.set_title("noisy image")
    ax5.axis("off")
    ax5.imshow(n_img, cmap="gray")

    ax6 = plt.subplot(2, 4, 4)
    ax6.set_title("corner response R")
    ax6.axis("off")
    ax6.imshow(r_n, cmap="gray")

    ax7 = plt.subplot(2, 4, 7)
    ax7.set_title("det(M)")
    ax7.axis("off")
    ax7.imshow(det_m_n, cmap="gray")

    ax8 = plt.subplot(2, 4, 8)
    ax8.set_title("trace(M)")
    ax8.axis("off")
    ax8.imshow(tra_m_n, cmap="gray")

    plt.tight_layout()
    plt.savefig("harris_response_visualization.png", dpi=150)
    plt.close()


def plot_harris_surface(r: np.ndarray, out_name: str, title: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = np.arange(r.shape[1])
    y = np.arange(r.shape[0])
    x_grid, y_grid = np.meshgrid(x, y)

    ax.plot_surface(x_grid, y_grid, r, rstride=20, cstride=20, alpha=0.3, cmap="viridis")
    ax.contour(x_grid, y_grid, r, zdir="z", offset=np.min(r))
    ax.contour(x_grid, y_grid, r, zdir="x", offset=np.min(x_grid))
    ax.contour(x_grid, y_grid, r, zdir="y", offset=np.max(y_grid))

    ax.set_xlabel("m")
    ax.set_ylabel("n")
    ax.set_zlabel("R(m,n)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()


def detect_corners(r: np.ndarray, threshold: Union[int, float] = 0.5) -> np.ndarray:
    corners_candidates = r > threshold
    return np.argwhere(corners_candidates)


def plot_corner_detection(img: np.ndarray, n_img: np.ndarray, corner_pos: np.ndarray, corner_pos_n: np.ndarray) -> None:
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, height_ratios=[8, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.set_title("corners in image")
    ax1.imshow(img, cmap="gray")
    ax1.plot(corner_pos[:, 1], corner_pos[:, 0], "r+", markersize=10, markeredgewidth=2)
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2.set_title("corners in noisy image")
    ax2.imshow(n_img, cmap="gray")
    ax2.plot(corner_pos_n[:, 1], corner_pos_n[:, 0], "r+", markersize=10, markeredgewidth=2)
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax3.text(0.2, 0.8, f"{len(corner_pos)} corners found", fontsize=14)
    ax3.axis("off")
    ax4.text(0.2, 0.8, f"{len(corner_pos_n)} corners found", fontsize=14)
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig("corner_detection.png")
    plt.close()


def rotate_img(img: np.ndarray, deg: Union[int, float] = 45, scale: Union[int, float] = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    return cv2.warpAffine(img, m, (w, h))


def rotation_test(img: np.ndarray, n_img: np.ndarray, threshold: float = 0.5) -> None:
    rotation_angles = [45, 60, 195]
    plt.figure(figsize=(15, 10))

    for idx, angle in enumerate(rotation_angles):
        img_rot = rotate_img(img, deg=angle, scale=1.0)
        fm_rot, fn_rot = compute_derivatives(img_rot, sigma=30)
        *_unused, _det_m_rot, _tra_m_rot, r_rot = compute_harris_corners(fm_rot, fn_rot, k=0.04, sigma=3)
        corner_pos_rot = detect_corners(r_rot, threshold=threshold)

        n_img_rot = rotate_img(n_img, deg=angle, scale=1.0)
        fm_n_rot, fn_n_rot = compute_derivatives(n_img_rot, sigma=30)
        *_unused2, _det_m_n_rot, _tra_m_n_rot, r_n_rot = compute_harris_corners(
            fm_n_rot, fn_n_rot, k=0.04, sigma=3
        )
        corner_pos_n_rot = detect_corners(r_n_rot, threshold=threshold)

        ax1 = plt.subplot(3, 2, 2 * idx + 1)
        ax1.set_title(f"Rotated {angle}° - noiseless ({len(corner_pos_rot)} corners)")
        ax1.imshow(img_rot, cmap="gray")
        ax1.plot(corner_pos_rot[:, 1], corner_pos_rot[:, 0], "r+", markersize=10, markeredgewidth=2)

        ax2 = plt.subplot(3, 2, 2 * idx + 2)
        ax2.set_title(f"Rotated {angle}° - noisy ({len(corner_pos_n_rot)} corners)")
        ax2.imshow(n_img_rot, cmap="gray")
        ax2.plot(corner_pos_n_rot[:, 1], corner_pos_n_rot[:, 0], "r+", markersize=10, markeredgewidth=2)

    plt.tight_layout()
    plt.savefig("rotation_test.png")
    plt.close()


def import_and_convert_img(img_path: str, img_size: int = 800) -> np.ndarray:
    path = Path(img_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Could not decode image: {path}")

    h, w = img.shape[:2]
    if h != img_size:
        scale_factor = img_size / h
        new_w = int(w * scale_factor)
        img = cv2.resize(img, (new_w, img_size), interpolation=cv2.INTER_LINEAR)

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_corners_improved(r: np.ndarray, threshold_ratio: float = 0.01, window_size: int = 5) -> np.ndarray:
    threshold = threshold_ratio * r.max()
    local_max = maximum_filter(r, size=window_size)
    corners = (r == local_max) & (r > threshold)
    return np.argwhere(corners)


def visualize_chessboard_harris(chessboard_path: str = "chessboard_crooked.png"):
    chess_img = import_and_convert_img(chessboard_path)
    fm_chess, fn_chess = compute_derivatives(chess_img, sigma=3)
    *_unused, det_m, tra_m, r = compute_harris_corners(fm_chess, fn_chess, k=0.04, sigma=3)
    corner_pos = detect_corners_improved(r, threshold_ratio=0.001, window_size=5)

    fig_combined = plt.figure(figsize=(14, 14))

    ax1 = plt.subplot(3, 2, 1)
    ax1.set_title("image", fontsize=14)
    ax1.axis("off")
    ax1.imshow(chess_img, cmap="gray")

    ax2 = plt.subplot(3, 2, 2)
    ax2.set_title("corner response R", fontsize=14)
    ax2.axis("off")
    ax2.imshow(r, cmap="gray")

    ax3 = plt.subplot(3, 2, 3)
    ax3.set_title("det(M)", fontsize=14)
    ax3.axis("off")
    ax3.imshow(det_m, cmap="gray")

    ax4 = plt.subplot(3, 2, 4)
    ax4.set_title("trace(M)", fontsize=14)
    ax4.axis("off")
    ax4.imshow(tra_m, cmap="gray")

    ax5 = plt.subplot(3, 1, 3)
    ax5.set_title("corners in image", fontsize=16)
    ax5.axis("off")
    ax5.imshow(chess_img, cmap="gray")
    ax5.plot(corner_pos[:, 1], corner_pos[:, 0], "ro", markersize=6, markeredgecolor="black", markeredgewidth=1)

    ax5.text(
        1.05,
        0.5,
        f"{len(corner_pos)} corners found",
        transform=ax5.transAxes,
        fontsize=14,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("chessboard_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    return chess_img, r, corner_pos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab06 Harris-corner analysis pipeline.")
    parser.add_argument("--noise-dev", type=float, default=30.0, help="Noise deviation for synthetic noisy image.")
    parser.add_argument(
        "--derivative-sigma",
        type=float,
        default=10.0,
        help="Sigma used for derivative computation in Harris experiments.",
    )
    parser.add_argument("--corner-threshold", type=float, default=0.5, help="Threshold for basic corner detector.")
    parser.add_argument(
        "--chessboard-path",
        default="chessboard_crooked.png",
        help="Path to chessboard image used for improved Harris visualization.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data_61 = ex61(3, 30)
        exercise061_plot(*data_61)

        data_62 = ex61(50, 3)
        exercise062_plot(*data_62)

        img = create_test_img(800, 800)
        n_img = add_noise(img, args.noise_dev)

        fm, fn = compute_derivatives(img, args.derivative_sigma)
        fm_n, fn_n = compute_derivatives(n_img, args.derivative_sigma)

        *_unused1, det_m, tra_m, r = compute_harris_corners(fm, fn, k=0.04, sigma=3)
        *_unused2, det_m_n, tra_m_n, r_n = compute_harris_corners(fm_n, fn_n, k=0.04, sigma=3)

        figure067(img, n_img, det_m, tra_m, r, det_m_n, tra_m_n, r_n)
        plot_harris_surface(r, "noiseless_harris_response.png", "Plot for R")
        plot_harris_surface(r_n, "noisy_harris_response.png", "Plot for R (noisy)")

        corner_pos = detect_corners(r, threshold=args.corner_threshold)
        corner_pos_n = detect_corners(r_n, threshold=args.corner_threshold)
        plot_corner_detection(img, n_img, corner_pos, corner_pos_n)

        rotation_test(img, n_img, threshold=args.corner_threshold)
        visualize_chessboard_harris(args.chessboard_path)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Lab06 analysis completed. Output figures saved in lab06 directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
