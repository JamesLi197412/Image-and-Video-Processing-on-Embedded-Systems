import cv2
import numpy as np

FEATURES = 400
MASK = None

img = cv2.imread('mandrill.png', 0)
if img is None:
    print("Error: Could not load image 'mandrill.png'. Check file path.")
    exit()

sift = cv2.SIFT_create(FEATURES)  # SIFT detector object
kp, des = sift.detectAndCompute(img, MASK)  # detect keypoints & compute descriptors

# Print the number of features found (length of the feature vector des)
print(f"Number of SIFT features detected: {len(kp)}")
print(f"Length of feature descriptor vector (des): {len(des)}")

feat_img = cv2.drawKeypoints(img, kp, None, (0, 0, 255), 4)  # keypoints on image.

cv2.imwrite("mandrillKP.png", feat_img)

