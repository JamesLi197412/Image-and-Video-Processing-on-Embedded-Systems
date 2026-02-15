# Lab 05 - Edge Detection Fundamentals

## Goal

Compare gradient-based edge operators and analyze Sobel/Laplace/Gaussian derivatives on synthetic and transformed chessboard images.

## Files

- `exe051.py`: basic gradient filters and visualization
- `exe053.py`: Sobel operators (SciPy + manual kernels)
- `exe054.py`: Gaussian derivative, Laplace, and LoG comparisons
- `chessboard.pbm`, `chessboard_crooked.png`: input images

## Run

```bash
cd lab05
python3 run_lab05.py
```

Legacy per-exercise wrappers are still available:

```bash
python3 exe051.py
python3 exe053.py
python3 exe054.py
```

## Expected Outputs

- `filtered.png`
- `filtered_crooked.png`
- `sobel_filtered.png`
- `laplace_filtered.png`
- `gaussian_smoothing_comparison.png`
- `gaussian_gradient_comparison.png`

## Dependencies

```bash
pip install numpy pillow matplotlib scipy opencv-python
```

## Notes

- Reusable operators now live in `edge_ops.py`.
- `run_lab05.py` is the recommended single entry point for the full lab output set.
