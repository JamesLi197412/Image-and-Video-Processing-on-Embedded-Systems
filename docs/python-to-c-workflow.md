# Python-To-C Optimization Workflow

This project is best presented as an engineering workflow, not just a Raspberry Pi class repo.

## Better Project Story

Stronger framing:

- prototype image-processing pipelines in Python
- isolate reusable kernels and performance-sensitive paths
- port selected kernels to portable C
- validate correctness on the host with repeatable tests
- prepare the code for later embedded deployment on Raspberry Pi

That shift highlights engineering judgment, systems thinking, and optimization discipline.

## What In This Repo Supports That Story

- `lab03` to `lab09` provide the Python prototyping and algorithm exploration layer.
- `csrc/` provides native C implementations for representative image kernels.
- `embedded_utils/native_image_ops.py` exposes the C kernels back to Python through `ctypes`.
- `benchmarks/benchmark_native_vs_python.py` measures correctness and timing on the host.
- `csrc/Makefile` supports host builds and native smoke tests.
- `README.md` now frames Raspberry Pi as a deployment context rather than the only development environment.
- `docs/embedded-roadmap.md` outlines the next steps toward a more production-style embedded pipeline.

## The Workflow To Emphasize

1. Use Python to explore the algorithm and confirm behavior quickly.
2. Choose the portion with the best payoff for native optimization.
3. Rebuild that portion in C with explicit data layout, bounds checking, and status codes.
4. Validate the native output on the host before binding or deploying it.
5. Keep the native module portable so it can later be wrapped from Python or cross-compiled for Raspberry Pi.



