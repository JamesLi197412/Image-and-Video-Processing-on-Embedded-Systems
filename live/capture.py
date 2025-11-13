import cv2

# Initialize the camera (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set camera properties (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


frame_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    # Add frame counter text to the display
    cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('MacBook Camera', frame)

    frame_count += 1

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if key == ord('q'):
        print("Quitting...")
        break

    # Press 's' to save snapshot
    elif key == ord('s'):
        filename = f'snapshot_{frame_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved as {filename}")

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed")