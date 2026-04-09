from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import subprocess
from typing import Tuple

import numpy as np


UInt8Ptr = ctypes.POINTER(ctypes.c_uint8)
FloatPtr = ctypes.POINTER(ctypes.c_float)


class NativeImageOpsUnavailableError(RuntimeError):
    """Raised when the native image-ops shared library is unavailable."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _csrc_dir() -> Path:
    return _repo_root() / "csrc"


def native_shared_library_path() -> Path:
    system = platform.system()
    suffix = {
        "Darwin": "dylib",
        "Linux": "so",
        "Windows": "dll",
    }.get(system)
    if suffix is None:
        raise NativeImageOpsUnavailableError(f"Unsupported platform for native image ops: {system}")
    return _csrc_dir() / "build" / f"libimage_ops.{suffix}"


def _native_sources() -> Tuple[Path, ...]:
    csrc_dir = _csrc_dir()
    return (
        csrc_dir / "image_ops.c",
        csrc_dir / "image_ops.h",
        csrc_dir / "Makefile",
    )


def ensure_native_shared_library(force_rebuild: bool = False) -> Path:
    lib_path = native_shared_library_path()

    should_build = force_rebuild or not lib_path.exists()
    if not should_build and lib_path.exists():
        lib_mtime = lib_path.stat().st_mtime
        should_build = any(src.exists() and src.stat().st_mtime > lib_mtime for src in _native_sources())

    if should_build:
        result = subprocess.run(
            ["make", "-C", str(_csrc_dir()), "shared"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise NativeImageOpsUnavailableError(
                "Failed to build native image ops shared library.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    if not lib_path.exists():
        raise NativeImageOpsUnavailableError(f"Native image ops shared library not found at {lib_path}")

    return lib_path


def native_image_ops_available() -> bool:
    try:
        load_native_image_ops()
    except NativeImageOpsUnavailableError:
        return False
    return True


def _configure_prototypes(lib: ctypes.CDLL) -> ctypes.CDLL:
    lib.csrc_convolve3x3_gray_u8.argtypes = [
        UInt8Ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        FloatPtr,
        FloatPtr,
        ctypes.c_size_t,
    ]
    lib.csrc_convolve3x3_gray_u8.restype = ctypes.c_int

    lib.csrc_sobel_gray_u8.argtypes = [
        UInt8Ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        FloatPtr,
        ctypes.c_size_t,
        FloatPtr,
        ctypes.c_size_t,
        FloatPtr,
        ctypes.c_size_t,
    ]
    lib.csrc_sobel_gray_u8.restype = ctypes.c_int

    lib.csrc_laplace_gray_u8.argtypes = [
        UInt8Ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        FloatPtr,
        ctypes.c_size_t,
    ]
    lib.csrc_laplace_gray_u8.restype = ctypes.c_int

    lib.csrc_harris_response_gray_u8.argtypes = [
        UInt8Ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        FloatPtr,
        ctypes.c_size_t,
    ]
    lib.csrc_harris_response_gray_u8.restype = ctypes.c_int

    lib.csrc_alpha_blend_bgra_over_bgr.argtypes = [
        UInt8Ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        UInt8Ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.csrc_alpha_blend_bgra_over_bgr.restype = ctypes.c_int

    return lib


@lru_cache(maxsize=1)
def load_native_image_ops() -> ctypes.CDLL:
    lib_path = ensure_native_shared_library()
    try:
        lib = ctypes.CDLL(str(lib_path))
    except OSError as exc:
        raise NativeImageOpsUnavailableError(f"Failed to load native image ops library: {exc}") from exc
    return _configure_prototypes(lib)


def _raise_on_status(status: int, op_name: str) -> None:
    if status == 0:
        return
    messages = {
        -1: "null pointer passed to native code",
        -2: "bad dimensions or invalid parameters",
        -3: "out-of-bounds blend placement",
        -4: "allocation failed inside native code",
    }
    detail = messages.get(status, "unknown native error")
    raise RuntimeError(f"{op_name} failed with status {status}: {detail}")


def _as_gray_u8(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"{name} expects a grayscale image with shape (H, W).")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_bgra_u8(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError(f"{name} expects a BGRA image with shape (H, W, 4).")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _as_bgr_u8(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} expects a BGR image with shape (H, W, 3).")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def convolve3x3_gray_u8(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    src = _as_gray_u8(image, name="convolve3x3_gray_u8")
    kernel_array = np.asarray(kernel, dtype=np.float32)
    if kernel_array.shape != (3, 3):
        raise ValueError("convolve3x3_gray_u8 expects a 3x3 kernel.")
    kernel_array = np.ascontiguousarray(kernel_array)

    dst = np.empty(src.shape, dtype=np.float32)
    lib = load_native_image_ops()
    status = lib.csrc_convolve3x3_gray_u8(
        src.ctypes.data_as(UInt8Ptr),
        ctypes.c_size_t(src.shape[1]),
        ctypes.c_size_t(src.shape[0]),
        ctypes.c_size_t(src.strides[0] // src.itemsize),
        kernel_array.ctypes.data_as(FloatPtr),
        dst.ctypes.data_as(FloatPtr),
        ctypes.c_size_t(dst.strides[0] // dst.itemsize),
    )
    _raise_on_status(status, "csrc_convolve3x3_gray_u8")
    return dst


def sobel_gray_u8(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = _as_gray_u8(image, name="sobel_gray_u8")
    grad_x = np.empty(src.shape, dtype=np.float32)
    grad_y = np.empty(src.shape, dtype=np.float32)
    magnitude = np.empty(src.shape, dtype=np.float32)

    lib = load_native_image_ops()
    status = lib.csrc_sobel_gray_u8(
        src.ctypes.data_as(UInt8Ptr),
        ctypes.c_size_t(src.shape[1]),
        ctypes.c_size_t(src.shape[0]),
        ctypes.c_size_t(src.strides[0] // src.itemsize),
        grad_x.ctypes.data_as(FloatPtr),
        ctypes.c_size_t(grad_x.strides[0] // grad_x.itemsize),
        grad_y.ctypes.data_as(FloatPtr),
        ctypes.c_size_t(grad_y.strides[0] // grad_y.itemsize),
        magnitude.ctypes.data_as(FloatPtr),
        ctypes.c_size_t(magnitude.strides[0] // magnitude.itemsize),
    )
    _raise_on_status(status, "csrc_sobel_gray_u8")
    return grad_x, grad_y, magnitude


def laplace_gray_u8(image: np.ndarray) -> np.ndarray:
    src = _as_gray_u8(image, name="laplace_gray_u8")
    dst = np.empty(src.shape, dtype=np.float32)

    lib = load_native_image_ops()
    status = lib.csrc_laplace_gray_u8(
        src.ctypes.data_as(UInt8Ptr),
        ctypes.c_size_t(src.shape[1]),
        ctypes.c_size_t(src.shape[0]),
        ctypes.c_size_t(src.strides[0] // src.itemsize),
        dst.ctypes.data_as(FloatPtr),
        ctypes.c_size_t(dst.strides[0] // dst.itemsize),
    )
    _raise_on_status(status, "csrc_laplace_gray_u8")
    return dst


def harris_response_gray_u8(image: np.ndarray, *, k: float = 0.04) -> np.ndarray:
    src = _as_gray_u8(image, name="harris_response_gray_u8")
    dst = np.empty(src.shape, dtype=np.float32)

    lib = load_native_image_ops()
    status = lib.csrc_harris_response_gray_u8(
        src.ctypes.data_as(UInt8Ptr),
        ctypes.c_size_t(src.shape[1]),
        ctypes.c_size_t(src.shape[0]),
        ctypes.c_size_t(src.strides[0] // src.itemsize),
        ctypes.c_float(k),
        dst.ctypes.data_as(FloatPtr),
        ctypes.c_size_t(dst.strides[0] // dst.itemsize),
    )
    _raise_on_status(status, "csrc_harris_response_gray_u8")
    return dst


def alpha_blend_bgra_over_bgr(overlay_bgra: np.ndarray, background_bgr: np.ndarray, *, x: int, y: int) -> np.ndarray:
    overlay = _as_bgra_u8(overlay_bgra, name="alpha_blend_bgra_over_bgr")
    background = _as_bgr_u8(background_bgr, name="alpha_blend_bgra_over_bgr").copy()

    lib = load_native_image_ops()
    status = lib.csrc_alpha_blend_bgra_over_bgr(
        overlay.ctypes.data_as(UInt8Ptr),
        ctypes.c_size_t(overlay.shape[1]),
        ctypes.c_size_t(overlay.shape[0]),
        ctypes.c_size_t(overlay.strides[0]),
        background.ctypes.data_as(UInt8Ptr),
        ctypes.c_size_t(background.shape[1]),
        ctypes.c_size_t(background.shape[0]),
        ctypes.c_size_t(background.strides[0]),
        ctypes.c_size_t(x),
        ctypes.c_size_t(y),
    )
    _raise_on_status(status, "csrc_alpha_blend_bgra_over_bgr")
    return background
