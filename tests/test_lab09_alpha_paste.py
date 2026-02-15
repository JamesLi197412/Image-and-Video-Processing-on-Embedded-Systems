import numpy as np

from lab09.alphachannel import add_alpha_channel
from lab09.pasteImage import paste_image


def test_add_alpha_channel_black_is_transparent():
    img = np.array(
        [
            [[0, 0, 0], [10, 20, 30]],
            [[0, 0, 0], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    bgra = add_alpha_channel(img)

    assert bgra.shape == (2, 2, 4)
    assert bgra[0, 0, 3] == 0
    assert bgra[1, 0, 3] == 0
    assert bgra[0, 1, 3] == 255
    assert bgra[1, 1, 3] == 255


def test_paste_image_alpha_blend():
    # Background is all black
    background = np.zeros((4, 4, 3), dtype=np.uint8)

    # Overlay is 2x2 red with full alpha
    overlay = np.zeros((2, 2, 4), dtype=np.uint8)
    overlay[:, :, 2] = 255  # red channel in BGR(A)
    overlay[:, :, 3] = 255  # alpha

    result = paste_image(overlay, background.copy(), x=1, y=1, w=2, h=2)

    # Overlay area should be red
    assert np.array_equal(result[1:3, 1:3, 2], np.full((2, 2), 255, dtype=np.uint8))
    # Blue/green channels stay zero
    assert np.array_equal(result[1:3, 1:3, 0], np.zeros((2, 2), dtype=np.uint8))
    assert np.array_equal(result[1:3, 1:3, 1], np.zeros((2, 2), dtype=np.uint8))
