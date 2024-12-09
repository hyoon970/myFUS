import cv2
import numpy as np
import cv2.aruco as aruco
import time

# Set the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)  # Correct method for getting dictionary
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_id = 23  # Change to the desired marker ID
marker_size = 0.0345  # Size of marker in meters (3.45 cm)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Variables to hold base tvec, rvec and relative positions
base_tvec = None
base_rvec = None
relative_positions = []  # List to store relative tvecs, rvecs of 4 positions

# Load camera calibration parameters (assume they're stored in files)
camera_matrix = np.load("camera_matrix_logi.npy")
dist_coeffs = np.load("dist_coeffs_logi.npy")

# Function to draw a closed box around the marker
def draw_marker_box(frame, corners, color=(0, 255, 0)):
    # Draw lines connecting the corners to form a closed box
    corners = np.int32(corners).reshape(-1, 2)
    for i in range(len(corners)):
        cv2.line(frame, tuple(corners[i]), tuple(corners[(i + 1) % len(corners)]), color, 2)

# Function to calculate relative transformation
def get_relative_pose(base_tvec, base_rvec, tvec, rvec):
    # Calculate the relative transformation (pose) between the base and the current position
    relative_tvec = tvec - base_tvec
    relative_rvec, _ = cv2.Rodrigues(rvec - base_rvec)
    return relative_tvec, relative_rvec

def detect_aruco_marker(frame):
    """
    Detect the ArUco marker in the frame, returning its corners, tvec and rvec if found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None and marker_id in ids:
        # Estimate pose of the marker
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
        # print("The position was really saved")
        return corners, tvec[0][0], rvec[0][0]  # Return the first marker's tvec and rvec
    return None, None, None


def main():
    global base_tvec, base_rvec

    num_positions_collected = 0
    # total_positions = 5  # Base + 4 relative positions

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        cv2.imshow("Initial viewing", frame)

        # # Detect the marker in the frame
        # corners, tvec, rvec = detect_aruco_marker(frame)
        #
        # if corners is not None:
        #     # Draw a box around the marker
        #     draw_marker_box(frame, corners[0])

        if base_tvec is None and base_rvec is None:
            frameb = frame.copy()
            cv2.putText(frame, "Please place the probe on your chest", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
            # Show current video frame with marker box
            cv2.imshow('Aruco Marker Detection', frameb)
            # Detect the marker in the frame
            corners, tvec, rvec = detect_aruco_marker(frameb)

            if corners is not None:
                # Draw a box around the marker
                draw_marker_box(frame, corners[0])
                # Save the base position (first marker location)
                base_tvec = tvec
                base_rvec = rvec
                print("Base position saved.")
                cv2.putText(frame, "Recording base position", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)
                # Show current video frame with marker box
                cv2.imshow('Aruco Marker Detection', frame)
                # time.sleep(5)

        elif num_positions_collected < 4 and key in [10, 13]:   # key 10, 13 is 'Enter'
            frameb = frame.copy()
            # Calculate and save the relative position
            relative_tvec, relative_rvec = get_relative_pose(base_tvec, base_rvec, tvec, rvec)
            relative_positions.append((relative_tvec, relative_rvec))
            num_positions_collected += 1
            print(f"Position {num_positions_collected} relative to base saved.")
            cv2.putText(frameb, f"Recording position {num_positions_collected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                        cv2.LINE_AA)
            print("Move it to the next position and press enter")
            cv2.imshow('Aruco Marker Detection', frameb)
            # time.sleep(5)

            # # Show current video frame with marker box
            # cv2.putText(frame, f"Recording position {num_positions_collected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 0, 0), 2,
            #             cv2.LINE_AA)
            # cv2.imshow('Aruco Marker Detection', frame)

        # Wait for user input
        key = cv2.waitKey(1)

        if key in [10, 13]:
            continue;
        else:
            cv2.putText(frame, f"Please move probe to position {num_positions_collected+1} and press enter", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2,
                        cv2.LINE_AA)
            cv2.imshow('Aruco Marker Detection', frame)

        # After 4 positions are collected, wait for 'Enter' key to display the boxes
        if num_positions_collected == 4 and key in [10, 13]:  # Enter key (10 for Linux/macOS, 13 for Windows)
            # Display all 5 locations by drawing a closed box around each one
            print("Displaying all marker positions...")

            print(base_tvec)
            print(base_rvec)
            # print(tvec_rel)
            # print(rvec_rel)

            # Draw base marker box
            ret, frame = cap.read()
            if not ret:
                break

            draw_marker_box(frame, corners[0], color=(0, 0, 255))  # Draw the base position in red

            # Loop through the 4 relative positions and draw boxes
            for i, (tvec_rel, rvec_rel) in enumerate(relative_positions):
                # Estimate new marker positions based on relative transformation
                projected_corners, _ = cv2.projectPoints(corners[0], rvec_rel, tvec_rel, camera_matrix, dist_coeffs)
                draw_marker_box(frame, projected_corners, color=(0, 255, 0))  # Green for other positions

            cv2.imshow('Aruco Marker Detection', frame)



        # Exit when 'q' is pressed
        if key == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
