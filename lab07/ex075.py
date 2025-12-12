import cv2
import numpy as np
from typing import List, Tuple, Optional

def draw_matches(img1: np.ndarray,
                 kp1: Tuple[cv2.KeyPoint, ...],
                 img2: np.ndarray,
                 kp2: Tuple[cv2.KeyPoint, ...],
                 matches: List[cv2.DMatch]) -> (
        np.ndarray):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    matches_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)


    matches_img[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    matches_img[:h2, w1:w1 + w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)


    for m in matches: # draw matches
        img1_idx = m.queryIdx
        img2_idx = m.trainIdx

        (m1, n1) = kp1[img1_idx].pt # coordinates
        (m2, n2) = kp2[img2_idx].pt

        cv2.circle(matches_img, (int(m1), int(n1)), 5, (0, 255, 0), 1)
        cv2.circle(matches_img, (int(m2) + w1 , int(n2)), 5, (0, 255, 0), 1)

        # Draw line connecting the two keypoints
        cv2.line(
            matches_img,
            (int(m1), int(n1)),
            (int(m2) + w1 , int(n2)),
            (0, 255, 0),
            1
        )

    return matches_img

FEATURES = 400
MASK = None

img1 = cv2.imread('Img1.png')  # Query image
img2 = cv2.imread('Img2.png')  # Train image

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

sift = cv2.SIFT_create(FEATURES)
kp1, des1 = sift.detectAndCompute(gray1, MASK)
kp2, des2 = sift.detectAndCompute(gray2, MASK)

bf = cv2.BFMatcher(cv2.NORM_L2)

matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
NNDR_RATIO = 0.75

for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair  # m is best match, n is second best
        if m.distance < NNDR_RATIO * n.distance:
            good_matches.append(m)

print(f"Good matches with Brute-Force found: {len(good_matches)}")

feat_img_a = cv2.drawKeypoints(img1, kp1, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
feat_img_b = cv2.drawKeypoints(img2, kp2, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imwrite('featImg1.png', feat_img_a)
cv2.imwrite('featImg2.png', feat_img_b)

result_matches = draw_matches(img1, kp1, img2, kp2, good_matches)
cv2.imwrite('matches.png', result_matches)


print(f"NNDR ratio used: {NNDR_RATIO}")
print(f"Number of good correspondences: {len(good_matches)}")



