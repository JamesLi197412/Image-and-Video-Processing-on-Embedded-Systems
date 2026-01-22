import cv2

# reference: https://docs.opencv.org/3.4.3/db/d5c/tutorial_py_bg_subtraction.html
def main():
    cap = cv2.VideoCapture(0)

    # Try MOG2 and KNN (OpenCV provides both)
    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=16, detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(history=200, dist2Threshold=400.0, detectShadows=True)

    use_knn = False  # option for mog2 or knn

    while True:
        ret, frame = cap.read()

        subtractor = knn if use_knn else mog2
        fgmask = subtractor.apply(frame)

        cv2.imshow("Frame", frame)
        cv2.imshow("FG Mask", fgmask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('t'):
            use_knn = not use_knn

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
