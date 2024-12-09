import cv2
import numpy as np
import cv2.aruco as aruco

# Initialize the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_id = 23  # Change to the desired marker ID
marker_size = 0.0345  # Size of marker in meters (3.45 cm)

def convert_grayscale_to_rgb_mask(grayscale_mask):
    # Create an RGB image by stacking the grayscale mask 3 times (initially, R=G=B)
    rgb_mask = cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR)

    # Create a binary mask where grayscale values are 255 (foreground)
    red_mask = grayscale_mask == 255

    # Set Red channel to 255 for the 255 pixels (leaving G and B as 0)
    rgb_mask[red_mask] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    return rgb_mask

def highlight_navy_blue(cap):

    ret, frame1 = cap.read()
    # if not ret:
    #     break
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    # Define the lower and upper range for navy-blue color in HSV
    # lower_navy_blue = np.array([100, 50, 50])   # Adjust the lower bound based on your needs
    # upper_navy_blue = np.array([130, 255, 150])  # Adjust the upper bound based on your needs

    # lower_navy_blue = np.array([30, 80, 70])   # Actually for yellow, adjust the lower bound based on your needs
    # upper_navy_blue = np.array([60, 150, 170])  # Actually for yellow, adjust the upper bound based on your needs

    # lower_navy_blue = np.array([121, 75, 85])   # Actually for bluegreen, adjust the lower bound based on your needs
    # upper_navy_blue = np.array([167, 245, 250])  # Actually for bluegreen, adjust the upper bound based on your needs

    # lower_navy_blue = np.array([121, 75, 85])   # Actually for bluegreen, adjust the lower bound based on your needs
    # upper_navy_blue = np.array([167, 245, 250])  # Actually for bluegreen, adjust the upper bound based on your needs

    # lower_navy_blue = np.array([150, 80, 40])   # Actually for red, adjust the lower bound based on your needs
    # upper_navy_blue = np.array([255, 245, 255])  # Actually for red, adjust the upper bound based on your needs

    lower_navy_blue = np.array([0, 63, 141])  # Actually for breastphantom on AS's desk, adjust the lower bound based on your needs
    upper_navy_blue = np.array([255, 255, 255])  # Actually for breastphantom on AS's desk, adjust the upper bound based on your needs

    # Create a mask for navy-blue pixels
    mask1 = cv2.inRange(hsv, lower_navy_blue, upper_navy_blue)

    # Create a new frame where navy-blue pixels will be highlighted in red
    highlighted_frame = frame1.copy()

    # Replace navy-blue pixels with bright red in the original frame
    highlighted_frame[mask1 != 0] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    rgb_mask = convert_grayscale_to_rgb_mask(mask1)

    return highlighted_frame, rgb_mask

# Function 1: Detect base position
def detect_base_position(cap, base_corners):
    # global base_tvec, base_rvec

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the ArUco marker
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        roi = frame[:, :]



        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        # If marker detected, draw a bounding box around it
        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
            # print(f'Frame type {type(frame)}')
            # print(f'Corners matrix {corners}')
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Display the frame with the bounding box
            cv2.imshow('Base Marker Detection', frame)
            # Wait for user input to save base position
            if cv2.waitKey(1) & 0xFF == ord('b'):
                # base_tvec = tvec[0][0]
                # base_rvec = rvec[0][0]
                # base_corners[0] = corners
                base_corners = corners
                print('Base position saved.')
                print(f'Base corners {base_corners}')
                break
        else:
            cv2.imshow('Base Marker Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Function 2: Detect relative positions
def detect_relative_positions(cap):
# def detect_relative_positions(cap, relative_corners):
#     global relative_positions

    positions_collected = 0

    while positions_collected < 4:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the ArUco marker
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        # If marker detected, draw a bounding box around it
        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
            aruco.drawDetectedMarkers(frame, corners, ids)
            #   aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            # Ask user to save the position
            cv2.imshow('Relative Marker Detection', frame)
            print(f"Press Enter to save position {positions_collected + 1}")

            if cv2.waitKey(1) & 0xFF == ord('\r'):  # '\r' is the Enter key
                relative_corners[positions_collected] = corners
                print(f"Position {positions_collected + 1} saved.")
                positions_collected += 1
        else:
            cv2.imshow('Relative Marker Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print(f'All saved corners {relative_corners}')

# Function to display the base and relative positions
def display_relative_positions(cap):

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        base_points = np.array(list(base_corners.values()), np.int32)

        # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
        base_points = base_points.reshape((-1, 1, 2))

        cv2.polylines(frame, [base_points], True, (0, 255, 0), 2)

        for i in range(4):
            cv2.polylines(frame, np.int32(relative_corners[i]), True, (255, 0, 0), 2)
            # Display the white frame
            # print(f'The corners are {relative_corners[i]}')

        cv2.imshow("Fit into this mask", frame)
        # cv2.imshow('Display Positions', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    global camera_matrix, dist_coeffs, relative_corners, base_corners, rgb_mask
    relative_corners = {}
    base_corners = {}

    # Load camera calibration parameters
    camera_matrix = np.load("camera_matrix_logi.npy")
    dist_coeffs = np.load("dist_coeffs_logi.npy")

    # Start video capture
    cap = cv2.VideoCapture(0)

    print('First we will record your current position and create a mask - try and stay within this mask!')
    # mask is acquired here
    highlighted_frame, rgb_mask = highlight_navy_blue(cap)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if ret:
            # Get the frame's dimensions (height, width, and channels)
            height, width, channels = frame.shape

            # Create a blank white frame with the same dimensions
            white_frame = np.ones((height, width, channels), dtype=np.uint8) * 255  # 255 for white color

        # this is the section where we superimpose the initially captured mask
        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)
        # Call the function to draw the bounding box and apply the mask
        # output_frame = draw_bounding_box_and_mask(roi, fgMask1)

        cv2.imshow('WebCam', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break

        # print("Press 'b' to detect base position, 'r' to detect relative positions, 'd' to display saved positions, "
        #       "or 'q' to quit.")

        print("Press 'r' to detect relative positions, 'd' to display saved positions, "
              "or 'q' to quit.")

        # Wait for key press
        key = cv2.waitKey(0000) & 0xFF  # This waits for key press

        if key == ord('b'):
            detect_base_position(cap, base_corners)
        elif key == ord('r'):
            # if base_tvec is not None and base_rvec is not None:
            # detect_relative_positions(cap, relative_corners)
            detect_relative_positions(cap)
            for i in range(4):
                print(f'the corner for the {i}th frame is {np.int32(relative_corners[i])}')
                cv2.polylines(white_frame, np.int32(relative_corners[i]), True, (255, 0, 0), 2)
                # Display the white frame
                cv2.imshow("White Frame", white_frame)
                # print(f'The corners are {relative_corners[i]}')
            # else:
            #     print("Please save the base position first.")
        elif key == ord('d'):
            print(f'lenght of relative positions array is {len(relative_corners)}')
            if len(relative_corners) == 4:
                display_relative_positions(cap)
            else:
                print("Please save the base position and all 4 relative positions first.")
        elif key == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()