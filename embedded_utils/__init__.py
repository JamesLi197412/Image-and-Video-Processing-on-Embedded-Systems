"""Shared utilities for embedded vision labs."""

from .metrics import RuntimeMetricsLogger
from .video_source import open_video_source, parse_resolution

__all__ = [
    "RuntimeMetricsLogger",
    "open_video_source",
    "parse_resolution",
]
