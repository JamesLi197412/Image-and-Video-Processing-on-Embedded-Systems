# Lab 04 - Image Processing and Histogram Analysis

## Goal

Practice grayscale conversion, channel analysis, histogram plotting, and histogram equalization on a sample image (`mandrill.png`).

## Files

- `lab42.py`: channel split visualization and image metadata display
- `ex42.py`: grayscale conversion (simple average and weighted)
- `ex43.py`: histogram implementation and plotting
- `ex44.py`: histogram equalization and summary figure
- `mandrill.png`: input image

## Run

Run each script from this folder:

```bash
cd lab04
python3 lab42.py --input mandrill.png
python3 ex42.py --input mandrill.png --output-dir .
python3 ex43.py --input mandrill.png --output "Histogram for mandrill.png"
python3 ex44.py --input mandrill.png --output "Plot resulting from the histogram equalization.png"
```

## Expected Outputs

- `mandrill_channels.png`
- `mandrill_simple.png`
- `mandrill_weighted.png`
- `Histogram for mandrill.png`
- `Plot resulting from the histogram equalization.png`

## Dependencies

```bash
pip install numpy pillow matplotlib scipy
```

## Notes

- Scripts assume `mandrill.png` is present in `lab04`.
- Plotting uses a non-interactive backend and writes image files directly.
- All scripts now support argparse and fail with clear messages on missing inputs.
