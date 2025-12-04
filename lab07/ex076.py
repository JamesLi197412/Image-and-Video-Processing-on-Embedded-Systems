import cv2
import numpy as np
from typing import List, Tuple, Optional
from lab07.ex075 import draw_matches


def stitch_2_images(img_a: np.ndarray, img_b: np.ndarray, h: np.ndarray) -> np.ndarray:
    assert all(isinstance(arr, np.ndarray) for arr in (img_a, img_b, h))

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    pts_a = np.float32([[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]).reshape(-1, 1, 2)
    pts_b = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)

    pts_b_pt = cv2.perspectiveTransform(pts_b, h)
    pts = np.concatenate((pts_a, pts_b_pt), axis=0)

    [m_min, n_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [m_max, n_max] = np.int32(pts.max(axis=0).ravel() + 0.5)

    ht = np.array([[1, 0, -m_min], [0, 1, -n_min], [0, 0, 1]])

    result = cv2.warpPerspective(img_b, ht.dot(h), (m_max - m_min, n_max - n_min))
    result[n_min:h_a + n_min, m_min:w_a + m_min] = img_a
    return result


def create_panorama(
        img1: np.ndarray,
        img2: np.ndarray,
        features: int = 400,
        mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert all(isinstance(arr, np.ndarray) for arr in (img1, img2))
    assert isinstance(features, int)
    assert mask is None or isinstance(mask, np.ndarray)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

    sift = cv2.SIFT_create(features)

    kp1, des1 = sift.detectAndCompute(gray1, mask)
    kp2, des2 = sift.detectAndCompute(gray2, mask)


    bf = cv2.BFMatcher(cv2.NORM_L2)

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    NNDR_RATIO = 0.75

    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < NNDR_RATIO * n.distance:
                good_matches.append(m)

    feat_img1 = cv2.drawKeypoints(img1, kp1, None, (0, 0, 255), 4)
    feat_img2 = cv2.drawKeypoints(img2, kp2, None, (0, 0, 255), 4)

    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    h, mask_h = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    result_matches = draw_matches(img1, kp1, img2, kp2, good_matches)

    result12 = stitch_2_images(img1, img2, h)
    return feat_img1, feat_img2, result_matches, result12

img1 = cv2.imread("imgNR1.png", 0)
img2 = cv2.imread("imgNR2.png", 0)

img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

feat_img1, feat_img2, result_matches, result12 = create_panorama(img1, img2)

cv2.imwrite("featImgA.png", feat_img1)
cv2.imwrite("featImgB.png", feat_img2)
cv2.imwrite("matches.png", result_matches)
cv2.imwrite("ImgAB.png", result12)


