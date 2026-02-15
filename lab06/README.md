# Lab 06 - Harris Corner Detection (Panorama Part 1)

## Goal

Implement and visualize Harris corner detection on synthetic and chessboard images, including noise sensitivity and rotation checks.

## File

- `lab06.py`: full experiment pipeline and plotting

## Run

```bash
cd lab06
python3 lab06.py
```

Optional parameters:

```bash
python3 lab06.py --noise-dev 30 --derivative-sigma 10 --corner-threshold 0.5 --chessboard-path chessboard_crooked.png
```

## Expected Outputs

- `Noiseless_Noisy.png`
- `eigen.png`
- `harris_response_visualization.png`
- `noiseless_harris_response.png`
- `noisy_harris_response.png`
- `corner_detection.png`
- `rotation_test.png`
- `chessboard_analysis.png`

## Dependencies

```bash
pip install numpy matplotlib scipy opencv-python
```

## Notes

- The script now runs only via explicit CLI entry (no import-time execution side effects).
- `chessboard_crooked.png` must exist in this folder.
