# Native C Image Ops

This directory adds a small host-portable C track to the project so you can practice and demonstrate a Python-to-C optimization workflow even without a Raspberry Pi on hand.

## What It Covers

- `csrc_convolve3x3_gray_u8`: grayscale 3x3 convolution
- `csrc_sobel_gray_u8`: Sobel X/Y gradients and magnitude
- `csrc_laplace_gray_u8`: Laplace edge response
- `csrc_harris_response_gray_u8`: Harris corner response using a compact 3x3 box window
- `csrc_alpha_blend_bgra_over_bgr`: BGRA-over-BGR alpha blending

These map naturally to the existing labs:

- `lab05`: convolution, Sobel, Laplace
- `lab06`: Harris response
- `lab09`: alpha blending

## Why This Strengthens The Project Narrative

Get practice with the parts of embedded work that matter on real projects:

- pointer arithmetic and image memory layout
- simple APIs with explicit status codes
- fixed-kernel implementations that are easy to profile
- host-side tests before hardware deployment
- a codebase that can later be cross-compiled for Raspberry Pi

This is the key shift in positioning:

- Python remains the fast algorithm-prototyping layer.
- C becomes the deliberate optimization layer.
- Hardware becomes a deployment target, not a blocker for progress.

## Build

```bash
make -C csrc
make -C csrc shared
make -C csrc test
```

The default build creates a static library in `csrc/build/`. The `shared` target creates a host-loadable shared library (`.dylib` on macOS, `.so` on Linux) that you can later call from Python with `ctypes` if you want to compare native and NumPy/OpenCV paths.

The repo now includes that bridge in `embedded_utils/native_image_ops.py`, plus a benchmark harness in `benchmarks/benchmark_native_vs_python.py`.


The flow in this project is:

The C code lives in image_ops.c and exposes functions declared in image_ops.h (line 19).
The shared library is built by the shared target in Makefile (line 31), which produces libimage_ops.dylib on macOS or .so on Linux.
Python loads that library in native_image_ops.py (line 150) using ctypes.CDLL(...).
The same file sets the C function signatures in _configure_prototypes(...) (line 86), so Python knows the argument and return types.
Wrapper functions like sobel_gray_u8(...) (line 222), laplace_gray_u8(...) (line 245), harris_response_gray_u8(...) (line 262), and alpha_blend_bgra_over_bgr(...) (line 280) convert NumPy arrays into contiguous buffers and pass raw pointers into C with .ctypes.data_as(...).
In the lab code, you opt into the native path like this:

sobel_filter(..., backend="c") (line 52)
laplace_filter(..., backend="c") (line 158)
paste_image(..., backend="c") (line 5)
compute_harris_response_native(...) (line 28)
So the real optimization path is:

Python/NumPy image -> ctypes wrapper -> compiled C shared library -> result back to NumPy