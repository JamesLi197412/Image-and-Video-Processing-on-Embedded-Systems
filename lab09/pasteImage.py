import cv2
import numpy as np


def paste_image(
    ag_img: np.ndarray,
    b_img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    backend: str = "python",
) -> np.ndarray:
    if ag_img.ndim != 3 or ag_img.shape[2] != 4:
        raise ValueError("paste_image expects ag_img as BGRA (H, W, 4).")
    if b_img.ndim != 3 or b_img.shape[2] != 3:
        raise ValueError("paste_image expects b_img as BGR (H, W, 3).")

    ag_img_resized = cv2.resize(ag_img, (w, h))

    if backend == "c":
        from embedded_utils.native_image_ops import alpha_blend_bgra_over_bgr

        blended = alpha_blend_bgra_over_bgr(ag_img_resized, b_img, x=x, y=y)
        b_img[...] = blended
        return b_img
    if backend != "python":
        raise ValueError(f"Unsupported backend: {backend}")

    alpha_ag = ag_img_resized[:, :, 3].astype(np.float32) / np.float32(255.0)
    alpha_b = np.float32(1.0) - alpha_ag

    for c in range(3):
        # out=α*overlay + (1−α)*background
        b_img[y : y + h, x : x + w, c] = alpha_ag * ag_img_resized[:, :, c] + alpha_b * b_img[y : y + h, x : x + w, c]

    return b_img
