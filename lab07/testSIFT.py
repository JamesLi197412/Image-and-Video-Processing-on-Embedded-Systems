from __future__ import annotations

import argparse

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick SIFT feature detection test.")
    parser.add_argument("--input", default="mandrill.png", help="Input image path.")
    parser.add_argument("--features", type=int, default=400, help="SIFT max features.")
    parser.add_argument("--output", default="mandrillKP.png", help="Output image path for keypoint visualization.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    img = cv2.imread(args.input, 0)
    if img is None:
        print(f"Error: Could not load image: {args.input}")
        return 1

    sift = cv2.SIFT_create(args.features)
    kp, des = sift.detectAndCompute(img, None)

    print(f"Number of SIFT keypoints detected: {len(kp)}")
    if des is not None:
        print(f"Descriptor matrix shape: {des.shape}")
    else:
        print("Descriptor matrix is None")

    feat_img = cv2.drawKeypoints(img, kp, None, (0, 0, 255), 4)
    cv2.imwrite(args.output, feat_img)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
