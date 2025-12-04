# Exercise 6.1
import numpy as np
import matplotlib as mpl
from scipy.optimize import fmin_ncg

mpl.use("agg")
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter
from typing import Union, Tuple

# 3D surface plots
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def create_test_img(width: int = 800, height: int = 800) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.float32)
    img[:] = 255   # background white
    img[0:400, 0:400] = 0   # black square
    return img

def add_noise(img: np.ndarray, dev: Union[int, float]) -> np.ndarray:
    assert isinstance(img, np.ndarray) and isinstance(dev, (int, float))
    noise = np.random.normal(loc=0, scale=dev, size=img.shape)
    n_img = img + noise
    n_img = np.clip(n_img, 0, 255)  # Clip to valid range
    return n_img.astype(np.float32)

def compute_derivatives( img: np.ndarray,sigma: Union[int, float] = 3) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(img, np.ndarray) and isinstance(sigma, (int, float))
    fm = gaussian_filter(img, sigma=(sigma, sigma), order=(0, 1))

    fn = gaussian_filter(img, sigma=(sigma, sigma), order=(1, 0))

    return fm, fn

def window_img(
    img_window: Tuple[int, int, int, int],
    img: np.ndarray,
    fm: np.ndarray,
    fn: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert (
        isinstance(img_window, tuple)
        and all(isinstance(t, int) for t in img_window)
        and all(isinstance(arr, np.ndarray) for arr in (img, fm, fn))
    )

    frame_size = 10               # red frame thickness
    red = np.array([255, 0, 0], dtype=np.uint8)
    m0, n0, m1, n1 = img_window   # window region

    # Convert to color so we can draw red frame
    img_with_frame = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Draw red frame (outer border of window)
    img_with_frame[m0:m0+frame_size, n0:n1] = red
    img_with_frame[m1-frame_size:m1, n0:n1] = red
    img_with_frame[m0:m1, n0:n0+frame_size] = red
    img_with_frame[m0:m1, n1-frame_size:n1] = red

    # Extract gradient patches
    w_fm = fm[m0:m1, n0:n1]
    w_fn = fn[m0:m1, n0:n1]

    return w_fm, w_fn, img_with_frame

def ex61(dev, sigma):
    img = create_test_img(800,800)
    n_img = add_noise(img, dev)

    fm, fn = compute_derivatives(img, sigma)
    fm_n, fn_n = compute_derivatives(n_img, sigma)

    flat = (600, 200, 700, 300)
    edge = (350, 100, 450, 200)
    corner = (350, 350, 450, 450)

    # noiseless image
    img_uint8 = img.astype(np.uint8)
    w_fm1, w_fn1, img1 = window_img(flat, img_uint8, fm, fn)
    w_fm2, w_fn2, img2 = window_img(edge, img_uint8, fm, fn)
    w_fm3, w_fn3, img3 = window_img(corner, img_uint8, fm, fn)

    # noisy image
    n_img_uint8 = np.clip(n_img, 0, 255).astype(np.float32)
    w_fm4, w_fn4, img4 = window_img(flat, img_uint8, fm_n, fn_n)
    w_fm5, w_fn5, img5 = window_img(edge, img_uint8, fm_n, fn_n)
    w_fm6, w_fn6, img6 = window_img(corner, img_uint8, fm_n, fn_n)

    return w_fm1, w_fn1, img1, w_fm2, w_fn2, img2, w_fm3, w_fn3, img3, w_fm4, w_fn4, img4, w_fm5, w_fn5, img5, w_fm6, w_fn6, img6, img, n_img

def exercise061_plot(w_fm1, w_fn1, img1, w_fm2, w_fn2, img2, w_fm3, w_fn3, img3,
    w_fm4, w_fn4, img4, w_fm5, w_fn5, img5, w_fm6, w_fn6, img6, img, n_img):
    # Create figure matching
    plt.figure(figsize=(18, 12))

    # Row 1: Original images
    plt.subplot(4, 6, 1), plt.title("noiseless"), plt.axis("off")
    plt.imshow(img, cmap='gray')

    plt.subplot(4, 6, 4), plt.title("noisy")
    plt.axis("off"), plt.imshow(n_img, cmap='gray')

    # Row 2: Flat region
    plt.subplot(4, 6, 7) ,plt.title("flat")
    plt.axis("off") , plt.imshow(img1)

    plt.subplot(4, 6, 8) , plt.title("Fm")
    plt.axis("off"),  plt.imshow(w_fm1, cmap='gray')

    plt.subplot(4, 6, 9), plt.title("Fn")
    plt.axis("off"),    plt.imshow(w_fn1, cmap='gray')

    plt.subplot(4, 6, 10)
    plt.title("noisy flat"),    plt.axis("off")
    plt.imshow(img4)

    plt.subplot(4, 6, 11),    plt.title("Fm")
    plt.axis("off"),    plt.imshow(w_fm4, cmap='gray')

    plt.subplot(4, 6, 12)
    plt.title("Fn"),    plt.axis("off")
    plt.imshow(w_fn4, cmap='gray')

    # Row 3: Edge region
    plt.subplot(4, 6, 13)
    plt.title("edge"),    plt.axis("off")
    plt.imshow(img2)

    plt.subplot(4, 6, 14)
    plt.title("Fm"),    plt.axis("off")
    plt.imshow(w_fm2, cmap='gray')

    plt.subplot(4, 6, 15)
    plt.title("Fn"),    plt.axis("off")
    plt.imshow(w_fn2, cmap='gray')

    plt.subplot(4, 6, 16)
    plt.title("noisy edge")
    plt.axis("off"),    plt.imshow(img5)

    plt.subplot(4, 6, 17)
    plt.title("Fm"),    plt.axis("off")
    plt.imshow(w_fm5, cmap='gray')

    plt.subplot(4, 6, 18)
    plt.title("Fn"),    plt.axis("off")
    plt.imshow(w_fn5, cmap='gray')

    # Row 4: Corner region
    plt.subplot(4, 6, 19)
    plt.title("corner"),    plt.axis("off")
    plt.imshow(img3)

    plt.subplot(4, 6, 20)
    plt.title("Fm"),    plt.axis("off")
    plt.imshow(w_fm3, cmap='gray')

    plt.subplot(4, 6, 21)
    plt.title("Fn"),    plt.axis("off")
    plt.imshow(w_fn3, cmap='gray')

    plt.subplot(4, 6, 22)
    plt.title("noisy corner")
    plt.axis("off"),    plt.imshow(img6)

    plt.subplot(4, 6, 23)
    plt.title("Fm"),    plt.axis("off")
    plt.imshow(w_fm6, cmap='gray')

    plt.subplot(4, 6, 24)
    plt.title("Fn"),    plt.axis("off")
    plt.imshow(w_fn6, cmap='gray')

    plt.tight_layout()
    plt.savefig("Noiseless_Noisy.png", dpi=150)
    plt.close()

def exercise062_plot(w_fm1, w_fn1, img1, w_fm2, w_fn2, img2, w_fm3, w_fn3, img3,
    w_fm4, w_fn4, img4, w_fm5, w_fn5, img5, w_fm6, w_fn6, img6, img, n_img):
    # Exercise 6.2
    plt.figure(figsize=(18, 12))
    plt.subplot(321), plt.title("flat")
    plt.ylabel("Fn")
    plt.axis([-10, 10, -10, 10])  # [Xmin, Xmax, Ymin, Ymax]
    plt.grid(True)
    plt.plot(w_fm1.flatten(), w_fn1.flatten(), "rx")

    # Subplot 322: noisy flat region
    plt.subplot(322), plt.title("noisy flat")
    plt.axis([-10, 10, -10, 10])
    plt.grid(True)
    plt.plot(w_fm4[:], w_fn4[:], "rx")

    # Subplot 323: edge region
    plt.subplot(323), plt.title("edge")
    plt.ylabel("Fn")
    plt.axis([-40, 40, -40, 40])
    plt.grid(True)
    plt.plot(w_fm2[:], w_fn2[:], "rx")

    # Subplot 324: noisy edge region
    plt.subplot(324), plt.title("noisy edge")
    plt.axis([-40, 40, -40, 40])
    plt.grid(True)
    plt.plot(w_fm5[:], w_fn5[:], "rx")

    # Subplot 325: corner region
    plt.subplot(325), plt.title("corner")
    plt.ylabel("Fn")
    plt.xlabel("Fm")
    plt.axis([-40, 40, -40, 40])
    plt.grid(True)
    plt.plot(w_fm3[:], w_fn3[:], "rx")

    # Subplot 326: noisy corner region
    plt.subplot(326), plt.title("noisy corner")
    plt.xlabel("Fm")
    plt.axis([-40, 40, -40, 40])
    plt.grid(True)
    plt.plot(w_fm6.flatten(), w_fn6.flatten(), "rx")

    plt.tight_layout()
    plt.savefig("eigen.png")
    plt.close()


w_fm1, w_fn1, img1, w_fm2, w_fn2, img2, w_fm3, w_fn3, img3, w_fm4, w_fn4, img4, w_fm5, w_fn5, img5, w_fm6, w_fn6, img6, img, n_img = ex61(3, 30)
exercise061_plot(w_fm1, w_fn1, img1, w_fm2, w_fn2, img2, w_fm3, w_fn3, img3, w_fm4, w_fn4, img4, w_fm5, w_fn5, img5, w_fm6, w_fn6, img6, img, n_img)

w_fm1, w_fn1, img1, w_fm2, w_fn2, img2, w_fm3, w_fn3, img3, w_fm4, w_fn4, img4, w_fm5, w_fn5, img5, w_fm6, w_fn6, img6, img, n_img = ex61(50, 3)
exercise062_plot(w_fm1, w_fn1, img1, w_fm2, w_fn2, img2, w_fm3, w_fn3, img3, w_fm4, w_fn4, img4, w_fm5, w_fn5, img5, w_fm6, w_fn6, img6, img, n_img)

def compute_harris_corners(
        fm: np.ndarray,
        fn: np.ndarray,
        k: Union[int, float] = 0.04,
        sigma: Union[int, float] = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert all(isinstance(arr, np.ndarray) for arr in (fm, fn)) and all(
        isinstance(num, (int, float)) for num in (k, sigma)
    )

    # Compute the elements of the structure tensor M
    # M = G_σ * [[Fm², Fm*Fn], [Fm*Fn, Fn²]]
    mmm = gaussian_filter(fm * fm, sigma)  # M_mm = G_σ * Fm²
    mmn = gaussian_filter(fm * fn, sigma)  # M_mn = G_σ * (Fm*Fn)
    mnm = mmn  # M_nm = M_mn (symmetric)
    mnn = gaussian_filter(fn * fn, sigma)  # M_nn = G_σ * Fn²

    # Compute determinant and trace of M
    # det(M) = M_mm * M_nn - M_mn * M_nm
    det_m = mmm * mnn - mmn * mnm

    # trace(M) = M_mm + M_nn
    tra_m = mmm + mnn

    # Harris corner response function
    # R = det(M) - k * trace(M)²
    r = det_m - k * (tra_m ** 2)

    return mmm, mmn, mnm, mnn, det_m, tra_m, r

img = create_test_img(800, 800)
n_img = add_noise(img, 30)

fm, fn = compute_derivatives(img, 10)
fm_n, fn_n = compute_derivatives(n_img, 10)

mmm, mmn, mnm, mnn,det_m, tra_m, r = compute_harris_corners(fm, fn, k=0.04, sigma=3)

# Compute Harris corners for noisy image
mmm, mmn, mnm, mnn,det_m_n, tra_m_n, r_n = compute_harris_corners(fm_n, fn_n, k=0.04, sigma=3)

def figure067(img, det_m, tra_m, r, det_m_n, tra_m_n, r_n):
    # Left side - Noiseless image
    # Row 1, Col 1: Original image
    ax1 = plt.subplot(2, 4, 1)
    ax1.set_title("image")
    ax1.axis("off")
    ax1.imshow(img, cmap='gray')

    # Row 1, Col 2: Corner response R
    ax2 = plt.subplot(2, 4, 2)
    ax2.set_title("corner response R")
    ax2.axis("off")
    ax2.imshow(r, cmap='gray')

    # Row 2, Col 1: Determinant
    ax3 = plt.subplot(2, 4, 5)
    ax3.set_title("det(M)")
    ax3.axis("off")
    ax3.imshow(det_m, cmap='gray')

    # Row 2, Col 2: Trace
    ax4 = plt.subplot(2, 4, 6)
    ax4.set_title("trace(M)")
    ax4.axis("off")
    ax4.imshow(tra_m, cmap='gray')

    # Right side - Noisy image
    # Row 1, Col 3: Noisy image
    ax5 = plt.subplot(2, 4, 3)
    ax5.set_title("nosy image")
    ax5.axis("off")
    ax5.imshow(n_img, cmap='gray')

    # Row 1, Col 4: Corner response R (noisy)
    ax6 = plt.subplot(2, 4, 4)
    ax6.set_title("corner response R")
    ax6.axis("off")
    ax6.imshow(r_n, cmap='gray')

    # Row 2, Col 3: Determinant (noisy)
    ax7 = plt.subplot(2, 4, 7)
    ax7.set_title("det(M)")
    ax7.axis("off")
    ax7.imshow(det_m_n, cmap='gray')

    # Row 2, Col 4: Trace (noisy)
    ax8 = plt.subplot(2, 4, 8)
    ax8.set_title("trace(M)")
    ax8.axis("off")
    ax8.imshow(tra_m_n, cmap='gray')

    plt.tight_layout()
    plt.savefig("harris_response_visualization.png", dpi=150)
    plt.close()

figure067(img, det_m, tra_m, r, det_m_n, tra_m_n, r_n)


# Plot for noiseless image
fig1 = plt.figure()
ax1 = fig1.add_subplot(projection="3d")

x = np.arange(r.shape[1])
y = np.arange(r.shape[0])
X, Y = np.meshgrid(x, y)
Z = r

ax1.plot_surface(X, Y, Z, rstride=20, cstride=20, alpha=0.3, cmap="viridis")
ax1.contour(X, Y, Z, zdir="z", offset=np.min(Z))
ax1.contour(X, Y, Z, zdir="x", offset=np.min(X))
ax1.contour(X, Y, Z, zdir="y", offset=np.max(Y))

ax1.set_xlabel("m")
ax1.set_ylabel("n")
ax1.set_zlabel("R(m,n)")
ax1.set_title("Plot for R")

plt.tight_layout()
plt.savefig("noiseless_harris_response.png")

# Plot for noisy image
fig2 = plt.figure()
ax2 = fig2.add_subplot(projection="3d")

x_n = np.arange(r_n.shape[1])
y_n = np.arange(r_n.shape[0])
X_n, Y_n = np.meshgrid(x_n, y_n)
Z_n = r_n

ax2.plot_surface(X_n, Y_n, Z_n, rstride=20, cstride=20, alpha=0.3, cmap="viridis")
ax2.contour(X_n, Y_n, Z_n, zdir="z", offset=np.min(Z_n))
ax2.contour(X_n, Y_n, Z_n, zdir="x", offset=np.min(X_n))
ax2.contour(X_n, Y_n, Z_n, zdir="y", offset=np.max(Y_n))

ax2.set_xlabel("m")
ax2.set_ylabel("n")
ax2.set_zlabel("R(m,n)")
ax2.set_title("Plot for R (noisy)")

plt.tight_layout()
plt.savefig("noisy_harris_response.png")


def detect_corners(r: np.ndarray, threshold: Union[int, float] = 0.5) -> np.ndarray:
    assert isinstance(r, np.ndarray) and isinstance(threshold, (int, float))

    # Find candidates where R > threshold (corner candidates)
    corners_candidates = r > threshold
    corner_pos = np.argwhere(corners_candidates)

    return corner_pos

corner_pos = detect_corners(r, threshold=0.5)
corner_pos_n = detect_corners(r_n, threshold=0.5)

# Create the visualization
fig = plt.figure()
gs = fig.add_gridspec(2, 2, height_ratios=[8, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Plot corners on noiseless image
ax1.set_title("corners in image")
ax1.imshow(img, cmap="gray")
ax1.plot(corner_pos[:, 1], corner_pos[:, 0], "r+", markersize=10, markeredgewidth=2)
ax1.set_xticks([])
ax1.set_yticks([])
for spine in ax1.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

# Plot corners on noisy image
ax2.set_title("corners in noisy image")
ax2.imshow(n_img, cmap="gray")
ax2.plot(corner_pos_n[:, 1], corner_pos_n[:, 0], "r+", markersize=10, markeredgewidth=2)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(True)
    spine.set_edgecolor("black")
    spine.set_linewidth(2)

# Text info for noiseless
ax3.text(0.2, 0.8, f"{len(corner_pos)} corners found", fontsize=14)
ax3.axis("off")

# Text info for noisy
ax4.text(0.2, 0.8, f"{len(corner_pos_n)} corners found", fontsize=14)
ax4.axis("off")

plt.tight_layout()
plt.savefig("corner_detection.png")


def rotate_img(
        img: np.ndarray,
        deg: Union[int, float] = 45,
        scale: Union[int, float] = 1.0
) -> np.ndarray:
    assert isinstance(img, np.ndarray) and all(
        isinstance(num, (int, float)) for num in (deg, scale)
    )

    h, w = img.shape[:2]
    # Get rotation matrix around center point
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    # Apply affine transformation
    return cv2.warpAffine(img, m, (w, h))


def rotation_test():
    rotation_angles = [45, 60, 195]

    # Create figure for rotation tests
    fig = plt.figure(figsize=(15, 10))

    for idx, angle in enumerate(rotation_angles):
        # Rotate noiseless image
        img_rot = rotate_img(img, deg=angle, scale=1.0)
        fm_rot, fn_rot = compute_derivatives(img_rot, sigma=30)
        mmm_rot, mmn_rot, mnm_rot, mnn_rot, det_m_rot, tra_m_rot, r_rot = compute_harris_corners(
            fm_rot, fn_rot, k=0.04, sigma=3
        )
        corner_pos_rot = detect_corners(r_rot, threshold=0.5)

        # Rotate noisy image
        n_img_rot = rotate_img(n_img, deg=angle, scale=1.0)
        fm_n_rot, fn_n_rot = compute_derivatives(n_img_rot, sigma=30)
        mmm_n_rot, mmn_n_rot, mnm_n_rot, mnn_n_rot, det_m_n_rot, tra_m_n_rot, r_n_rot = compute_harris_corners(
            fm_n_rot, fn_n_rot, k=0.04, sigma=3
        )
        corner_pos_n_rot = detect_corners(r_n_rot, threshold=0.5)

        # Plot noiseless rotated
        ax1 = plt.subplot(3, 2, 2 * idx + 1)
        ax1.set_title(f"Rotated {angle}° - noiseless ({len(corner_pos_rot)} corners)")
        ax1.imshow(img_rot, cmap="gray")
        ax1.plot(corner_pos_rot[:, 1], corner_pos_rot[:, 0], "r+", markersize=10, markeredgewidth=2)

        # Plot noisy rotated
        ax2 = plt.subplot(3, 2, 2 * idx + 2)
        ax2.set_title(f"Rotated {angle}° - noisy ({len(corner_pos_n_rot)} corners)")
        ax2.imshow(n_img_rot, cmap="gray")
        ax2.plot(corner_pos_n_rot[:, 1], corner_pos_n_rot[:, 0], "r+", markersize=10, markeredgewidth=2)


    plt.tight_layout()
    plt.savefig("rotation_test.png")

rotation_test()


def import_and_convert_img(img_path: str, img_size: int = 800) -> np.ndarray:
    assert isinstance(img_path, str) and isinstance(img_size, int)

    # Read image
    img = cv2.imread(img_path)
    # Get original dimensions
    h, w = img.shape[:2]

    if h != img_size:
        # Calculate scale factor to make height = img_size
        scale_factor = img_size / h

        # Calculate new width maintaining aspect ratio
        new_w = int(w * scale_factor)
        new_h = img_size

        # Resize image
        img = cv2.resize(
            img,
            (new_w, new_h),  # (width, height) for cv2.resize
            interpolation=cv2.INTER_LINEAR,
        )

    # Convert to grayscale
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



def detect_corners_improved(r: np.ndarray, threshold_ratio: float = 0.01, window_size: int = 5) -> np.ndarray:
    """
    Improved corner detection with adaptive thresholding and non-maximum suppression
    """
    from scipy.ndimage import maximum_filter

    # Adaptive threshold based on maximum R value
    threshold = threshold_ratio * r.max()

    # Non-maximum suppression
    local_max = maximum_filter(r, size=window_size)
    corners = (r == local_max) & (r > threshold)

    corner_pos = np.argwhere(corners)
    return corner_pos

def visualize_chessboard_harris():
    """Create complete Harris corner detection visualization for chessboard"""

    # Load and convert chessboard image
    chessboard_path = "chessboard_crooked.png"
    chess_img = import_and_convert_img(chessboard_path)

    fm_chess, fn_chess = compute_derivatives(chess_img, sigma=3)

    # Compute Harris corner response
    mmm, mmn, mnm, mnn, det_m, tra_m, r = compute_harris_corners(fm_chess, fn_chess, k=0.04, sigma=3)

    # Detect corners using improved method
    corner_pos = detect_corners_improved(r, threshold_ratio=0.001, window_size=5)


    # Create the visualization (2 rows, 2 columns + text area)
    fig = plt.figure(figsize=(14, 10))

    # Row 1, Col 1: Original chessboard image
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("image", fontsize=14)
    ax1.axis("off")
    ax1.imshow(chess_img, cmap='gray', vmin=0, vmax=255)

    #  Corner response R
    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("corner response R", fontsize=14)
    ax2.axis("off")
    # Normalize R for better visualization
    r_display = r.copy()
    r_display[r_display < 0] = 0  # Clip negative values
    ax2.imshow(r_display, cmap='gray')

    # Determinant
    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("det(M)", fontsize=14)
    ax3.axis("off")
    det_display = det_m.copy()
    det_display[det_display < 0] = 0  # Clip negative values
    ax3.imshow(det_display, cmap='gray')

    # Trace
    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("trace(M)", fontsize=14)
    ax4.axis("off")
    ax4.imshow(tra_m, cmap='gray')

    plt.tight_layout()
    #plt.savefig("chessboard_harris_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    #print("Harris analysis saved: chessboard_harris_analysis.png")

    # Create second figure: Detected corners
    fig2 = plt.figure(figsize=(10, 10))

    ax5 = plt.subplot(1, 1, 1)
    ax5.set_title("corners in image", fontsize=16)
    ax5.axis("off")
    ax5.imshow(chess_img, cmap='gray', vmin=0, vmax=255)

    # Plot detected corners as red circles with black centers
    ax5.plot(corner_pos[:, 1], corner_pos[:, 0], 'ro',
             markersize=8, markerfacecolor='red',
             markeredgecolor='black', markeredgewidth=1.5)

    # Add text annotation
    ax5.text(0.02, 0.98, f"{len(corner_pos)} corners found",
             transform=ax5.transAxes, fontsize=14,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    #plt.savefig("chessboard_corners_detected.png", dpi=150, bbox_inches='tight')
    plt.close()
    #print("Corner detection saved: chessboard_corners_detected.png")

    # Create combined figure like in reference
    fig_combined = plt.figure(figsize=(14, 14))

    # Top row
    ax1 = plt.subplot(3, 2, 1)
    ax1.set_title("image", fontsize=14)
    ax1.axis("off")
    ax1.imshow(chess_img, cmap='gray')

    ax2 = plt.subplot(3, 2, 2)
    ax2.set_title("corner response R", fontsize=14)
    ax2.axis("off")
    ax2.imshow(r, cmap='gray')

    # Middle row
    ax3 = plt.subplot(3, 2, 3)
    ax3.set_title("det(M)", fontsize=14)
    ax3.axis("off")
    ax3.imshow(det_m, cmap='gray')

    ax4 = plt.subplot(3, 2, 4)
    ax4.set_title("trace(M)", fontsize=14)
    ax4.axis("off")
    ax4.imshow(tra_m, cmap='gray')

    # Bottom row - spanning both columns
    ax5 = plt.subplot(3, 1, 3)
    ax5.set_title("corners in image", fontsize=16)
    ax5.axis("off")
    ax5.imshow(chess_img, cmap='gray')
    ax5.plot(corner_pos[:, 1], corner_pos[:, 0], 'ro',
             markersize=6, markerfacecolor='red',
             markeredgecolor='black', markeredgewidth=1)

    # Add text annotation in the right area
    ax5.text(1.05, 0.5, f"{len(corner_pos)} corners found",
             transform=ax5.transAxes, fontsize=14,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("chessboard_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


    return chess_img, r, corner_pos

# Run the complete chessboard analysis
chess_img, r_chess, corners_chess = visualize_chessboard_harris()


