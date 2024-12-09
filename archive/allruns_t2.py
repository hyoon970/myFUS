import cv2
import cv2.aruco as aruco
import numpy as np

# Initialize the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_id = 23  # Change to the desired marker ID
marker_size = 0.0345  # Size of marker in meters (3.45 cm)

# # Variables to store the base position and relative positions
# base_tvec = None
# base_rvec = None
# relative_positions = []

def convert_grayscale_to_rgb_mask(grayscale_mask):
    # Create an RGB image by stacking the grayscale mask 3 times (initially, R=G=B)
    rgb_mask = cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR)

    # Create a binary mask where grayscale values are 255 (foreground)
    red_mask = grayscale_mask == 255

    # Set Red channel to 255 for the 255 pixels (leaving G and B as 0)
    rgb_mask[red_mask] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    return rgb_mask

def detect_silhouette(capture):
    global rgb_mask
    backSub = cv2.createBackgroundSubtractorMOG2()

    i = 0

    totalframes = 20

    # frame1 = np.zeros((480, 640))
    fgMask1 = np.zeros((480, 640))

    while i < totalframes:
        ret, frame = capture.read()
        if frame is None:
            break

        fgMask = backSub.apply(frame)

        cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv2.imshow('Frame', frame)
        cv2.imshow('FG Mask', fgMask)
        fgMask[fgMask < 128] = 0
        fgMask1 = fgMask1 + fgMask

        i = i + 1

        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    fgMask1 = fgMask1 / totalframes
    fgMask1[fgMask1 < 20] = 0  # this threshold seems to work for getting rid of background pixels even with complex
    # backgrounds
    ret, mask = cv2.threshold(fgMask1.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)
    # print(f'size of mask is {np.shape(mask)} and type of data is {type(mask[1,1])}')
    rgb_mask = convert_grayscale_to_rgb_mask(mask)

def display_mask(capture):
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        # copy over frame
        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function 1: Detect base position
def detect_base_position(cap, base_corners):
    global base_tvec, base_rvec
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the ArUco marker
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

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
                base_tvec = tvec[0][0]
                base_rvec = rvec[0][0]
                base_corners[0] = corners
                print('Base position saved.')
                print(f'Base corners {base_corners}')
                break
        else:
            cv2.imshow('Base Marker Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Function 2: Detect relative positions
def detect_relative_positions(cap, relative_corners):
    global relative_positions

    positions_collected = 0

    while positions_collected < 4:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect the ArUco marker
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

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

        base_points = np.array(list(base_corners.values()), np.int32)

        # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
        base_points = base_points.reshape((-1, 1, 2))

        cv2.polylines(frame, [base_points], True, (0, 255, 0), 1)

        for i in range(4):
            cv2.polylines(frame, np.int32(relative_corners[i]), True, (255, 0, 0), 1)
            # Display the white frame
            # print(f'The corners are {relative_corners[i]}')

        cv2.imshow("Fit into this mask", frame)
        # cv2.imshow('Display Positions', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Main function to capture video and execute the detection functions
def main():
    global camera_matrix, dist_coeffs, relative_corners, base_corners
    relative_corners = {}
    base_corners = {}

    # Load camera calibration parameters
    camera_matrix = np.load("camera_matrix_logi.npy")
    dist_coeffs = np.load("dist_coeffs_logi.npy")

    # Start video capture
    cap = cv2.VideoCapture(0)

    print('First we will record your current position and create a mask - try and stay within this mask!')
    detect_silhouette(cap)

    print("Press 'b' to detect base position, 'r' to detect relative positions, 'd' to display saved positions, "
          "or 'q' to quit.")

    while True:
        ret, frame = cap.read()  # Read frame from the video capture
        if not ret:
            break  # Break if the video capture fails

        # Display the current video frame in a window
        cv2.imshow('Frame', frame)

        if ret:
            # Get the frame's dimensions (height, width, and channels)
            height, width, channels = frame.shape

            # Create a blank white frame with the same dimensions
            white_frame = np.ones((height, width, channels), dtype=np.uint8) * 255  # 255 for white color

        # Wait for key press
        key = cv2.waitKey(0000) & 0xFF  # This waits for key press

        if key == ord('b'):
            detect_base_position(cap, base_corners)
        elif key == ord('r'):
            if base_tvec is not None and base_rvec is not None:
                detect_relative_positions(cap, relative_corners)
                for i in range(4):
                    print(f'the corner for the {i}th frame is {np.int32(relative_corners[i])}')
                    cv2.polylines(white_frame, np.int32(relative_corners[i]), True, (255, 0, 0), 1)
                    # Display the white frame
                    cv2.imshow("White Frame", white_frame)
                    # print(f'The corners are {relative_corners[i]}')
            else:
                print("Please save the base position first.")
        elif key == ord('d'):
            print(f'lenght of relative positions array is {len(relative_corners)}')
            if len(relative_corners) == 4:
                display_relative_positions(cap)
            else:
                print("Please save the base position and all 4 relative positions first.")
        elif key == ord('q'):
            break

        # Release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
