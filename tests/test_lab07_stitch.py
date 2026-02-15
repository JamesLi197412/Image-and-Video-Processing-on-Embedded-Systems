import numpy as np

from lab07.ex076 import stitch_2_images


def test_stitch_identity_overlays_first_image():
    img_a = np.zeros((20, 20, 3), dtype=np.uint8)
    img_b = np.zeros((20, 20, 3), dtype=np.uint8)

    img_a[:, :] = [0, 0, 255]  # red in BGR
    img_b[:, :] = [0, 255, 0]  # green in BGR

    h_identity = np.eye(3, dtype=np.float32)

    result = stitch_2_images(img_a, img_b, h_identity)

    assert result.shape[0] >= 20 and result.shape[1] >= 20
    # img_a is pasted on top for overlap area
    assert np.array_equal(result[0, 0], np.array([0, 0, 255], dtype=np.uint8))
