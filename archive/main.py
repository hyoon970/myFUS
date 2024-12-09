
import cv2
import numpy as np
import cv2.aruco as aruco
# from getkey import getkey, keys


# # importing homemade files
# import fingerdetect
# import arucodetect

import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Initialize run variables
current_run = 1
total_runs = 4
run_text = f"Run {current_run}"


def execute_function(run_number):
    """
    Simulate the execution of a function.
    Here you can add any operation you want to perform for each run.
    """
    print(f"Executing function for {run_number}")


def main():
    global current_run, run_text

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture video frame.")
            break

        if current_run == 0:
            # Display the current run text on the video frame
            run_text = f"Please place the aruco marker on the centre of your chest"
            cv2.putText(frame, run_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Show the frame with the run text
            cv2.imshow('Video Capture', frame)

            # Wait for the user's keypress
            key = cv2.waitKey(1)

            # Check if the "Enter" key was pressed (ASCII code 13 for Windows or 10 for macOS/Linux)
            if key in [10, 13]:  # Enter key
                # Execute the function for the current run
                detect_marker(frame)

        else:
            # Display the current run text on the video frame
            run_text = f"Place probe in position {current_run}"
            cv2.putText(frame, run_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Show the frame with the run text
            cv2.imshow('Video Capture', frame)

            # Wait for the user's keypress
            key = cv2.waitKey(1)

            # Check if the "Enter" key was pressed (ASCII code 13 for Windows or 10 for macOS/Linux)
            if key in [10, 13]:  # Enter key
                # Execute the function for the current run
                execute_function(run_text)

        # Move to the next run
        current_run += 1

        # Exit the loop when all runs are completed
        if current_run > total_runs:
                print("All runs completed.")
                break

        # Exit on pressing the 'q' key
        if key == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Constants
MARKER_ID = 23  # The ArUco marker ID we want to detect
MARKER_SIZE = 0.0345  # Marker size in meters (3.45 cm)

# Load camera calibration data (assumed previously calibrated)
camera_matrix = np.load("../camera_matrix.npy")  # Replace with your file path
dist_coeffs = np.load("../dist_coeffs.npy")  # Replace with your file path

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