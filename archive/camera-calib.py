import cv2
import numpy as np
import glob


def calibrate_camera(chessboard_size=(12, 11), square_size=0.0125, num_frames=20, display_corners=False):
    """
    Calibrates the webcam using a checkerboard pattern.

    :param chessboard_size: Size of the checkerboard (number of interior corners per row and column).
    :param square_size: Size of a square in the checkerboard (in meters).
    :param num_frames: Number of frames to capture for calibration.
    :param display_corners: If True, displays detected corners on the checkerboard.
    :return: Camera matrix and distortion coefficients.
    """

    # Prepare object points (3D points in real world space)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # Scale by the actual size of the squares

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    print("frame capture hasn't started yet")

    # Capture frames from the webcam
    cap = cv2.VideoCapture(0)
    captured_frames = 0

    while captured_frames < num_frames:
        print("you are now in the frame capture loop")
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            if display_corners:
                cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                cv2.imshow('Chessboard', frame)

            captured_frames += 1
            print(f"Captured frame {captured_frames}/{num_frames}")

        cv2.imshow('Webcam', frame)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calibrate the camera
    print("Calibrating camera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the camera matrix and distortion coefficients
    if ret:
        print("Camera calibration successful!")
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)
        return camera_matrix, dist_coeffs
    else:
        print("Camera calibration failed.")
        return None, None

if __name__ == "__main__":
    # Calibrate the camera using a 12x13 checkerboard and squares of size 1.25 cm
    print("we will now start camera calibration")
    camera_matrix, dist_coeffs = calibrate_camera(chessboard_size=(12, 11), square_size=0.0125, num_frames=20, display_corners=True)

    # If needed, save the results
    if camera_matrix is not None and dist_coeffs is not None:
        np.save("camera_matrix_logi.npy", camera_matrix)
        np.save("dist_coeffs_logi.npy", dist_coeffs)
        print("Camera matrix and distortion coefficients saved to files.")