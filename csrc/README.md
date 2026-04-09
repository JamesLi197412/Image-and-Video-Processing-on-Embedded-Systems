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

You still get practice with the parts of embedded work that matter on real projects:

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

## Suggested Next Step

After this foundation, the best follow-up is to benchmark the same bridge on Raspberry Pi hardware and compare host-side versus on-device behavior. That turns the repo from "I wrote some C too" into "I measure where native optimization actually pays off."
