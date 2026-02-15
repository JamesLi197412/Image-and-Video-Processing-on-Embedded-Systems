# Lab 07 - Panorama Pipeline (SIFT and Stitching)

## Goal

Capture overlapping frames and build a panorama pipeline using SIFT keypoints, NNDR matching, homography estimation, and image warping.

## Files

- `ex071.py`: capture one test image from Picamera2
- `ex072.py`: capture multiple frames via GPIO buttons
- `ex073.py`: manual Gaussian/DoG scale-space exploration
- `testSIFT.py`: quick SIFT keypoint count test
- `ex075.py`: feature matching visualization for two images
- `ex076.py`: two-image panorama stitching
- `ex077.py`: end-to-end capture + stitching driver

## Hardware/Runtime Requirements

- Raspberry Pi camera (for capture scripts)
- GPIO buttons (green on pin `5`, red on pin `6`)
- Python packages: `picamera2`, `gpiozero`, `opencv-python`, `numpy`, `psutil`

## Typical Runs

```bash
cd lab07
python3 testSIFT.py --input mandrill.png
python3 ex073.py --input mandrill.png --levels 5 --sigma 1.6
python3 ex075.py --img1 Img1.png --img2 Img2.png
python3 ex076.py --img1 imgNR1.png --img2 imgNR2.png
```

For live capture:

```bash
cd lab07
python3 ex072.py --source auto --frame-count 4
python3 ex077.py --capture --debug-memory
```

## Expected Outputs (depends on input files)

- `test_image_<w>_<h>.png`
- `mandrillKP.png`
- `featImg1.png`, `featImg2.png`, `matches.png`
- `featImgA.png`, `featImgB.png`, `ImgAB.png`
- `panorama.png`

## Notes

- Some scripts require local input images (`Img1.png`, `Img2.png`, `imgNR1.png`, `imgNR2.png`) that are not committed in this folder.
- Run scripts from `lab07` so relative paths resolve correctly.
- Realtime scripts now emit CSV runtime metrics under `metrics/`.
