# Lab 03 - Intro to Image Signal Processing (Pillow)

## Goal

Learn basic image creation and manipulation in Python using Pillow, including checkerboard generation, simple transforms, and timing comparisons.

## Files

- `imgprocess.py`: core lab implementation
- `chessboard.pbm`, `chessboard_3.png`: sample outputs/assets

## Main Functions in `imgprocess.py`

- `create_chessboard_1()`: pixel-by-pixel checkerboard
- `create_chessboard_2()`: block/rectangle-based checkerboard
- `create_chessboard_3()`: point-drawing checkerboard
- `create_black_frame(pic, frame_width)`: add border
- `transpose_pic(pic)`: transpose image

## Run Examples

From this folder:

```bash
cd lab03
python3 - <<'PY'
from imgprocess import create_chessboard_2, create_chessboard_3, save_image
img = create_chessboard_2()
save_image(img, "chessboard_2", "png")
create_chessboard_3()
print("Generated chessboard images.")
PY
```

## Expected Outputs

- `chessboard_2.png`
- `chessboard_3.png`

## Dependencies

- `Pillow`

```bash
pip install pillow
```

## Notes

- Functions are designed as importable helpers; this lab does not have one central CLI entry point.
- The script writes output files in the current working directory.
