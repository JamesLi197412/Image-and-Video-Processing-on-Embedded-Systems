import numpy as np

from lab09.grabcut import grab_cut


def test_grabcut_returns_same_shape_and_dtype():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img[30:70, 30:70] = [0, 0, 0]

    result = grab_cut(img, x=20, y=20, w=60, h=60, iter_count=1)

    assert result.shape == img.shape
    assert result.dtype == img.dtype
