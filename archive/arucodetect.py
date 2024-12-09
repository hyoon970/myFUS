import cv2
import numpy as np
import cv2.aruco as aruco

# Constants
MARKER_ID = 23  # The ArUco marker ID we want to detect
MARKER_SIZE = 0.0345  # Marker size in meters (3.45 cm)

# Load camera calibration data (assumed previously calibrated)
camera_matrix = np.load("camera_matrix.npy")  # Replace with your file path
dist_coeffs = np.load("dist_coeffs.npy")  # Replace with your file path

# Set the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Correct method for getting dictionary
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

def detect_marker(frame):
    """
    Detect the ArUco marker, estimate its position and orientation, and display it.
    :param frame: The video frame to process.
    :return: None
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    #corners, ids, rejected = detector.detectMarkers(gray, aruco_dict, parameters=parameters)
    corners, ids, rejected = detector.detectMarkers(gray)

    # If markers are detected
    if ids is not None and MARKER_ID in ids:
        # Index of the marker with ID 23
        index = np.where(ids == MARKER_ID)[0][0]

        # Estimate the pose of the marker
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[index], MARKER_SIZE, camera_matrix, dist_coeffs)

        # Extract rotation and translation vectors
        rvec = rvec[0][0]
        tvec = tvec[0][0]

        # Draw the detected marker and the axis for its orientation
        aruco.drawDetectedMarkers(frame, corners)  # Draw bounding box around marker
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)  # Draw axis (length=5cm)

        # Calculate distance to the marker (Euclidean distance)
        distance = np.linalg.norm(tvec)

        # Print the marker details
        print(f"Detected ArUco marker ID {MARKER_ID}")
        print(f"Distance: {distance:.2f} meters")
        print(f"Translation Vector (tvec): {tvec}")
        print(f"Rotation Vector (rvec): {rvec}")

        # Display distance on the frame
        cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Marker not detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('ArUco Marker Detection', frame)

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Detect marker and estimate pose
        detect_marker(frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
