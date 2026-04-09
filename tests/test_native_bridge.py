import numpy as np

from embedded_utils.native_image_ops import ensure_native_shared_library
from lab05.edge_ops import laplace_filter, sobel_filter
from lab06.lab06 import compute_harris_response_native
from lab09.pasteImage import paste_image


def test_native_sobel_matches_python_backend():
    ensure_native_shared_library()
    rng = np.random.default_rng(10)
    img = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)

    py_fm, py_fn, py_mag, _ = sobel_filter(img, backend="python")
    c_fm, c_fn, c_mag, _ = sobel_filter(img, backend="c")

    assert np.allclose(py_fm, c_fm, atol=1e-4)
    assert np.allclose(py_fn, c_fn, atol=1e-4)
    assert np.allclose(py_mag, c_mag, atol=1e-4)


def test_native_laplace_matches_python_backend():
    ensure_native_shared_library()
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)

    py_laplace = laplace_filter(img, backend="python")
    c_laplace = laplace_filter(img, backend="c")

    assert np.allclose(py_laplace, c_laplace, atol=1e-4)


def test_native_harris_response_bridge_returns_corner_signal():
    ensure_native_shared_library()
    img = np.zeros((16, 16), dtype=np.uint8)
    img[8:, 8:] = 255

    response = compute_harris_response_native(img)

    assert response.shape == img.shape
    assert response.max() > 0
    assert response[8, 8] > response[0, 0]


def test_paste_image_native_backend_matches_python_backend():
    ensure_native_shared_library()
    rng = np.random.default_rng(12)
    background = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    overlay = rng.integers(0, 256, size=(6, 6, 4), dtype=np.uint8)

    py_result = paste_image(overlay, background.copy(), x=4, y=5, w=6, h=6, backend="python")
    c_result = paste_image(overlay, background.copy(), x=4, y=5, w=6, h=6, backend="c")

    assert np.array_equal(py_result, c_result)
