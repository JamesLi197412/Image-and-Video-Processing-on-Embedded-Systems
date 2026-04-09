"""Shared utilities for embedded vision labs."""

from .metrics import RuntimeMetricsLogger
from .native_image_ops import (
    NativeImageOpsUnavailableError,
    alpha_blend_bgra_over_bgr,
    convolve3x3_gray_u8,
    ensure_native_shared_library,
    harris_response_gray_u8,
    laplace_gray_u8,
    native_image_ops_available,
    native_shared_library_path,
    sobel_gray_u8,
)
from .video_source import open_video_source, parse_resolution

__all__ = [
    "RuntimeMetricsLogger",
    "NativeImageOpsUnavailableError",
    "alpha_blend_bgra_over_bgr",
    "convolve3x3_gray_u8",
    "ensure_native_shared_library",
    "harris_response_gray_u8",
    "laplace_gray_u8",
    "native_image_ops_available",
    "native_shared_library_path",
    "open_video_source",
    "parse_resolution",
    "sobel_gray_u8",
]
