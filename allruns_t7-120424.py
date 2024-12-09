import cv2
import numpy as np
import cv2.aruco as aruco
from sklearn.cluster import KMeans
from collections import Counter

# Initialize the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_id = 23  # Change to the desired marker ID
marker_size = 0.0345  # Size of marker in meters (3.45 cm)


def get_dominant_color_hsv(cap, k=2):
    ret, image = cap.read()
    image = cv2.flip(image, 1)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_pixels = hsv_image.reshape((-1, 3))

    # KMeans with faster initialization and fewer clusters
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
    kmeans.fit(hsv_pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    label_counts = Counter(labels)
    dominant_color = colors[label_counts.most_common(1)[0][0]]
    lower_skin = np.array([max(0, dominant_color[0] - 20), 70, 100], dtype=np.uint8)
    upper_skin = np.array([min(179, dominant_color[0] + 20), 255, 255], dtype=np.uint8)
    print(f'color1 is {lower_skin}')
    print(f'color1 is {upper_skin}')
    # lower_skin = np.array([(dominant_color[0] - 20), 20, 50], dtype=np.uint8)
    # upper_skin = np.array([(dominant_color[0] + 20), 255, 255], dtype=np.uint8)

    # return dominant_color
    return lower_skin, upper_skin


def convert_grayscale_to_rgb_mask2(grayscale_mask):
    # Create an RGB image by stacking the grayscale mask 3 times (initially, R=G=B)
    rgb_mask = cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR)

    # Create a binary mask where grayscale values are 255 (foreground)
    red_mask = grayscale_mask == 255

    # Set Red channel to 255 for the 255 pixels (leaving G and B as 0)
    rgb_mask[red_mask] = [0, 255, 0]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    return rgb_mask

def convert_grayscale_to_rgb_mask(grayscale_mask):
    # Create an RGB image by stacking the grayscale mask 3 times (initially, R=G=B)
    rgb_mask = cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR)

    # Create a binary mask where grayscale values are 255 (foreground)
    red_mask = grayscale_mask == 255

    # Set Red channel to 255 for the 255 pixels (leaving G and B as 0)
    rgb_mask[red_mask] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    return rgb_mask

def highlight_navy_blue(cap, lower_skin, upper_skin):

    ret, frame1 = cap.read()

    frame1 = cv2.flip(frame1, 1)

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

    # lower_navy_blue = np.array([0, 63, 141])  # Actually for breastphantom on AS's desk, adjust the lower bound based on your needs
    # upper_navy_blue = np.array([255, 255, 255])  # Actually for breastphantom on AS's desk, adjust the upper bound based on your needs

    # Create a mask for navy-blue pixels
    # mask1 = cv2.inRange(hsv, lower_navy_blue, upper_navy_blue)
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    # Create a new frame where navy-blue pixels will be highlighted in red
    highlighted_frame = frame1.copy()

    # Replace navy-blue pixels with bright red in the original frame
    highlighted_frame[mask1 != 0] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    rgb_mask = convert_grayscale_to_rgb_mask(mask1)

    return highlighted_frame, rgb_mask

# Function 2: Detect relative positions
def detect_base_position(cap, base_flag, base_corners, base_width):

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[:, :]
        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        if base_flag == 0:
            # Detect the ArUco marker
            corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
            print(f"the corners are {corners}")
            # print(f"type and dimension {type(corners)} {size(corners)}")
            # break

            # If marker detected, draw a bounding box around it
            if ids is not None:
                # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
                aruco.drawDetectedMarkers(frame, corners, ids)

                # Ask user to save the position
                cv2.imshow('Base Marker', frame)
                print(f"Press Enter to save base position")

                if cv2.waitKey(1) & 0xFF == ord('\r'):  # '\r' is the Enter key
                    base_corners = corners
                    # Extract the 2D array of coordinates
                    coords = base_corners[0][0]

                    # Calculate distances between consecutive corners (with wrapping to first corner)
                    side_lengths = [np.linalg.norm(coords[i] - coords[(i + 1) % len(coords)]) for i in
                                    range(len(coords))]

                    # Calculate the average side length
                    average_side_length = np.mean(side_lengths)

                    print("Average side length:", average_side_length)
                    base_width = average_side_length
                    print("Base position saved.")
                    base_flag = 1
                    break
            else:
                cv2.imshow('Base Marker Detection', frame)
                print("Please place the aruco marker as flat as possible on your sternum")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        return base_corners, base_width

def compare_base_position(cap, base_corners, base_width):
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[:, :]
        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        #draw original base square
        # base_points = np.array(base_corners[0][0], np.int32)
        print(f"base corners are {base_corners}")
        base_points = base_corners[0][0]

        # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
        base_points = base_points.reshape((-1, 1, 2))

        cv2.polylines(frame, [base_points], True, (255, 0, 0), 2)

        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        aruco.drawDetectedMarkers(frame, corners, ids)

        # calculate the new side length
        coords = corners[0][0]

        # Calculate distances between consecutive corners (with wrapping to first corner)
        side_lengths_current = [np.linalg.norm(coords[i] - coords[(i + 1) % len(coords)]) for i in
                                range(len(coords))]

        # Calculate the average side length
        average_side_length_current = np.mean(side_lengths_current)

        print("Average side length:", average_side_length_current)
        aruco_width_current = average_side_length_current

        if aruco_width_current > base_width+5:
            print("Move further away from the camera")
        elif aruco_width_current < base_width-5:
            print("Move towards the camera")
        else:
            print("This position is adequate, stay here")
            # draw original base square
            current_points = np.array(corners[0][0], np.int32)

            # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
            current_points = current_points.reshape((-1, 1, 2))

            cv2.polylines(frame, [current_points], True, (0, 255, 0), 4)

        cv2.imshow("Base positioning step", frame)


    print(f'All saved corners {relative_corners}')


# Function 2: Detect relative positions
def detect_relative_positions(cap):

    positions_collected = 0

    while positions_collected < total_positions:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        # Detect the ArUco marker
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        # If marker detected, draw a bounding box around it
        if ids is not None:
            # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
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
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)

        relative_points = np.array(list(relative_corners.values()), np.int32)

        # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
        relative_points = relative_points.reshape((-1, 1, 2))

        cv2.polylines(frame, [relative_points], True, (0, 255, 0), 2)

        for i in range(total_positions):
            cv2.polylines(frame, np.int32(relative_corners[i]), True, (255, 0, 0), 2)

        cv2.imshow("Fit into this mask", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def compare_probe_overlap(cap):

    tolerance = 5

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
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

        for i in range(total_positions):
            cv2.polylines(frame, np.int32(relative_corners[i]), True, (255, 0, 0), 2)

        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        aruco.drawDetectedMarkers(frame, corners, ids)

        # check overlap
        if ids is not None:
            print(f'corners of marker are {corners}')
            print(f'corners of saved position are {relative_corners[0]}')
            for i in range(total_positions):
                diff = np.array(corners) - np.array(relative_corners[i])
                print(f'difference array is {diff}')
                if diff.all() <= 10:
                    overlap_id = i
                    break
            # Draw the filled polygon
            fill_color = (0, 255, 0)  # Fill color in BGR (Green in this case)
            # cv2.fillPoly(frame, [np.array(relative_corners[overlap_id])], fill_color)
            cv2.polylines(frame, np.int32(relative_corners[overlap_id]), True, (0, 255, 0), 4)

        cv2.imshow("Keep positioning it into the frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    global camera_matrix, dist_coeffs, relative_corners, base_corners, base_width, rgb_mask, total_positions
    total_positions = 1
    relative_corners = {}
    base_corners = []
    base_width = 0
    base_flag = 0

    # Load camera calibration parameters
    camera_matrix = np.load("camera_matrix_logi.npy")
    dist_coeffs = np.load("dist_coeffs_logi.npy")

    # Start video capture
    cap = cv2.VideoCapture(0)

    print('First we will record your current position and create a mask - try and stay within this mask!')
    # mask is acquired here
    lower_skin, upper_skin = get_dominant_color_hsv(cap, k=2)
    highlighted_frame, rgb_mask = highlight_navy_blue(cap, lower_skin, upper_skin)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if frame is None:
            break

        if ret:
            # Get the frame's dimensions (height, width, and channels)
            height, width, channels = frame.shape

            # Create a blank white frame with the same dimensions
            white_frame = np.ones((height, width, channels), dtype=np.uint8) * 255  # 255 for white color

        print(f'height and width is {height} {width}')

        # this is the section where we superimpose the initially captured mask
        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)


        cv2.imshow('WebCam', frame)


        print("Press 'b' for base position calibration, 'r' to detect relative positions, 'd' to display saved positions, 'c' for probe repositioning, "
              "or 'q' to quit.")

        # Wait for key press
        key = cv2.waitKey(0000) & 0xFF  # This waits for key press

        if key == ord('b'):
            base_corners, base_width = detect_base_position(cap, base_flag, base_corners, base_width)
        if key == ord('l'): # l for line-up
            compare_base_position(cap, base_corners, base_width)
        if key == ord('r'):
            detect_relative_positions(cap)
            for i in range(total_positions):
                print(f'the corner for the {i}th frame is {np.int32(relative_corners[i])}')
                cv2.polylines(white_frame, np.int32(relative_corners[i]), True, (255, 0, 0), 2)
                # Display the white frame
                cv2.imshow("White Frame", white_frame)
        elif key == ord('d'):
            print(f'lenght of relative positions array is {len(relative_corners)}')
            if len(relative_corners) == total_positions:
                display_relative_positions(cap)
            else:
                print("Please save the base position and all 4 relative positions first.")
        elif key == ord('c'):
            compare_probe_overlap(cap)
        elif key == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()