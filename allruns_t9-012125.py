import cv2
import numpy as np
import cv2.aruco as aruco
from sklearn.cluster import KMeans
from collections import Counter
import time

# Initialize the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_id = 23  # Change to the desired marker ID (maybe from each marker pattern)
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


def convert_grayscale_to_rgb_mask2(grayscale_mask): # make a green mask
    # Create an RGB image by stacking the grayscale mask 3 times (initially, R=G=B)
    rgb_mask = cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR)

    # Create a binary mask where grayscale values are 255 (foreground)
    red_mask = grayscale_mask == 255

    # Set Red channel to 255 for the 255 pixels (leaving G and B as 0)
    rgb_mask[red_mask] = [0, 255, 0]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    return rgb_mask

def convert_grayscale_to_rgb_mask(grayscale_mask): # make a red mask
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

    # Create a mask for navy-blue pixels
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

        # draw original base square
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

################################
def calculate_overlap_percentage(img1,img2):
    # Ensure the two images have the same dimensions
    assert img1.shape == img2.shape, "Images must have the same dimensions"

    # Calculate the intersection (logical AND)
    intersection = np.logical_and(img1, img2).astype(np.int8)

    # Calculate the union (logical OR)
    union = np.logical_or(img1, img2).astype(np.uint8)

    #Count the non-zero pixels in intersection and union
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)

    if union_count == 0:
        print("Reference mask was not detected, please start it again")
        return 0

    # Calculate the overlap percentage
    overlap_percentage = (intersection_count / union_count) * 100
    overlap_percentage = round(overlap_percentage, 2)

    return overlap_percentage


def calculate_mask_overlap(cap, mask, lower_skin, upper_skin):  # 'mask' should be rgb mask
    positions_collected = 0
    count = 0


    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  ## This is the frame we see
        # Detect the ArUco marker
        #corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        ###

        #cap_trans = cv2.VideoCapture(0)

        ret_trans, frame1_trans = cap.read()

        frame1_trans = cv2.flip(frame1_trans, 1)

        ## mask with background subtraction
        # Convert the frame into grayscale
        gray = cv2.cvtColor(frame1_trans, cv2.COLOR_BGR2GRAY)
        # Create a backgroundsubtractor
        back_sub = cv2.createBackgroundSubtractorMOG2()
        # Apply backgroundsubtractor
        fg_mask = back_sub.apply(gray)
        # Threshold to get binary mask
        _, binary_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame1_trans, contours, -1, (0, 255, 0), 2)

        if count == 1:
            cv2.imwrite('test-mask/contour_mask.png', fg_mask)


        # Convert the frame to HSV color space
        hsv_trans = cv2.cvtColor(frame1_trans, cv2.COLOR_BGR2HSV)

        # Define the lower and upper range for navy-blue color in HSV
        # lower_navy_blue = np.array([100, 50, 50])   # Adjust the lower bound based on your needs
        # upper_navy_blue = np.array([130, 255, 150])  # Adjust the upper bound based on your needs

        # Create a mask for navy-blue pixels
        mask1_trans = cv2.inRange(hsv_trans, lower_skin, upper_skin)

        # This is for saving the mask_trans for checking if it is working properly
        # cv2.imwrite(f'test-mask/mask_{count}.png', mask1_trans)

        hsv_from_ref = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2HSV)

        # Define the lower and upper range for navy-blue color in HSV
        # lower_navy_blue = np.array([100, 50, 50])   # Adjust the lower bound based on your needs
        # upper_navy_blue = np.array([130, 255, 150])  # Adjust the upper bound based on your needs

        # Create a mask for navy-blue pixels
        mask_ref = cv2.inRange(hsv_from_ref, lower_skin, upper_skin) ## This is a binary image

        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0  # 'mask' should be rgb mask
        roi += rgb_mask.astype(np.uint8)


        if count == 0:
            print(mask1_trans.shape)
            print('\n')
            print(mask_ref.shape)

        count += 1

        print(count)

        overlap_percent = calculate_overlap_percentage(mask_ref,mask1_trans)
        print('overlap_percentage: ' + str(overlap_percent) + '%')

        if count == 1:
            cv2.imwrite('test_mask/mask_ref.png', mask_ref)
            cv2.imwrite('test_mask/mask1_trans.png', mask1_trans)


        # Define the text to display
        text = 'overlap_percentage: ' + str(overlap_percent) + '%'

        # Define the font, position, font size, color, and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20,20)
        font_scale = 0.5
        color = (255,0,0)
        thickness = 2

        # Add the text to the image
        #cv2.putText(frame, text, position, font, font_scale, color, thickness)
        cv2.putText(frame1_trans, text, position, font, font_scale, color, thickness)

        #cv2.imshow('Mask', frame)
        cv2.imshow('Mask', frame1_trans)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def calculate_mask_overlap_contour(cap, mask):  # 'mask' should be rgb mask
    count = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  ## This is the frame we see

        ret_trans, frame1_trans = cap.read()

        frame1_trans = cv2.flip(frame1_trans, 1)

        ## mask with background subtraction (background should be white)
        # Convert the frame into grayscale
        gray = cv2.cvtColor(frame1_trans, cv2.COLOR_BGR2GRAY)

        ####### I changed the threshold from 200 to 190 #######
        # Threshold to get binary mask
        _, binary_mask_trans = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY) # should adjust the range according to the background


        ######################################
        ####### what I added on 250121 #######
        height, width = binary_mask_trans.shape[:2]
        center_column_index = width // 2
        ratio_to_keep_from_center = 0.5
        binary_mask_trans[:, :round(center_column_index * (1 - ratio_to_keep_from_center))] = 0
        binary_mask_trans[:, round(center_column_index * (1 + ratio_to_keep_from_center)):] = 0
        ######################################
        ######################################

        # Find contours
        contours, _ = cv2.findContours(binary_mask_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print(type(contours))
        #print(contours)
        cv2.drawContours(frame1_trans, contours, -1, (0, 255, 0), 2)



        if count == 1:
            cv2.imwrite('test_mask/contour_mask.png', mask) ## To check if the mask is properly made


        ### Change binary mask into rgb mask
        # Create an empty HSV image
        hsv_image_contour = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        # Define HSV values for white and black pixels
        # Hue: 0-179, Saturation: 0-255, Value: 0-255
        hsv_foreground_contour = (60, 255, 255)  # Example: green
        hsv_background_contour = (0, 0, 0)  # Black
        # Apply the HSV values based on the binary image
        hsv_image_contour[mask == 255] = hsv_foreground_contour
        hsv_image_contour[mask == 0] = hsv_background_contour
        # Convert HSV to BGR for visualization or saving
        bgr_mask_contour = cv2.cvtColor(hsv_image_contour, cv2.COLOR_HSV2BGR)

        ######################################
        ####### what I added on 250121 #######

        height, width = bgr_mask_contour.shape[:2]
        center_column_index = width // 2
        ratio_to_keep_from_center = 0.5
        bgr_mask_contour[:, :round(center_column_index*(1-ratio_to_keep_from_center))] = 0
        bgr_mask_contour[:, round(center_column_index * (1+ratio_to_keep_from_center)):] = 0

        mask[:, :round(center_column_index*(1-ratio_to_keep_from_center))] = 0
        mask[:, round(center_column_index * (1+ratio_to_keep_from_center)):] = 0
        ######################################
        ######################################

        ### Attach the mask on the frame
        roi = frame1_trans[:, :]
        # Set an index of where the mask is
        roi[np.where(bgr_mask_contour)] = 0  # 'mask' should be rgb mask
        roi += bgr_mask_contour.astype(np.uint8)

        if count == 0:
            print(binary_mask_trans.shape)
            print('\n')
            print(mask.shape)


        # Save roi as an image
        #cv2.imwrite(f'roi_{count}.png', roi)
        count += 1

        print(count)

        overlap_percent = calculate_overlap_percentage(binary_mask_trans,mask)
        #overlap_percent = calculate_overlap_percentage(binary_mask_trans, mask)
        print('overlap_percentage: ' + str(overlap_percent) + '%')

        if count == 1:
            cv2.imwrite('test_mask/mask_ref.png', binary_mask_trans)
            cv2.imwrite('test_mask/mask1_trans.png', mask)


        # Define the text to display
        text = 'overlap_percentage: ' + str(overlap_percent) + '%'

        # Define the font, position, font size, color, and thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20,20)
        font_scale = 0.5
        color = (255,0,0) # Blue, (B, G, R)
        thickness = 2

        ### Indicating the direction we should move
        # Calculate the total number of pixels
        total_pixels_ref = mask.size
        total_pixels_trans = binary_mask_trans.size
        # Count the number of pixels with value 0
        zero_pixels_ref = np.sum(mask == 0)
        zero_pixels_trans = np.sum(binary_mask_trans == 0)
        # Calculate the percentage of 0 pixels
        percentage_of_zeros_ref = (zero_pixels_ref / total_pixels_ref) * 100
        percentage_of_zeros_trans = (zero_pixels_trans / total_pixels_trans) * 100

        percentage_threshold = 95
        if overlap_percent >= percentage_threshold:
            text0 = 'Aligned!'
            # Define the font, position, font size, color, and thickness
            font0 = cv2.FONT_HERSHEY_SIMPLEX
            position0 = (20, 50)
            font_scale0 = 0.5
            color0 = (255, 0, 0)
            thickness0 = 2
            cv2.putText(frame1_trans, text0, position0, font0, font_scale0, color0, thickness0)
        if (percentage_of_zeros_trans >= percentage_of_zeros_ref) and overlap_percent < percentage_threshold:
            text1 = 'Please move backward'
            # Define the font, position, font size, color, and thickness
            font1 = cv2.FONT_HERSHEY_SIMPLEX
            position1 = (20, 50)
            font_scale1 = 0.5
            color1 = (255, 0, 0)
            thickness1 = 2
            cv2.putText(frame1_trans, text1, position1, font1, font_scale1, color1, thickness1)
        if (percentage_of_zeros_trans < percentage_of_zeros_ref) and overlap_percent < percentage_threshold:
            text2 = 'Please move forward'
            # Define the font, position, font size, color, and thickness
            font2 = cv2.FONT_HERSHEY_SIMPLEX
            position2 = (20, 50)
            font_scale2 = 0.5
            color2 = (255, 0, 0)
            thickness2 = 2
            cv2.putText(frame1_trans, text2, position2, font2, font_scale2, color2, thickness2)



        # Add the text to the image
        cv2.putText(frame1_trans, text, position, font, font_scale, color, thickness)

        # Show the image on the screen
        cv2.imshow('Mask', frame1_trans)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
################################################################
################################################################


# Function 2: Detect relative positions
def detect_relative_positions(cap):

    positions_collected = 0

    while positions_collected < total_positions:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1) ## This is the frame we see
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


def compare_probe_overlap(cap): ## We need to modify this into mask overlap

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

    ## mask with background subtraction (background should be white)
    time.sleep(5) ## To avoid flash lighting at the first time
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    # Convert the frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold to get binary mask
    _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # should adjust the range according to the background


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


        print("Press 'm' for the mask alignment, 'b' for base position calibration, 'r' to detect relative positions, 'd' to display saved positions, 'c' for probe repositioning, "
              "or 'q' to quit.")

        # Wait for key press
        key = cv2.waitKey(0000) & 0xFF  # This waits for key press

        if key == ord('b'):
            base_corners, base_width = detect_base_position(cap, base_flag, base_corners, base_width)
        if key == ord('l'): # l for line-up
            compare_base_position(cap, base_corners, base_width)

        ## make another ord function
        if key == ord('m'):
            #calculate_mask_overlap(cap, rgb_mask, lower_skin, upper_skin)
            calculate_mask_overlap_contour(cap, binary_mask)
        if key == ord('r'):
            detect_relative_positions(cap)
            for i in range(total_positions):
                print(f'the corner for the {i}th frame is {np.int32(relative_corners[i])}')
                cv2.polylines(white_frame, np.int32(relative_corners[i]), True, (255, 0, 0), 2)
                # Display the white frame
                cv2.imshow("White Frame", white_frame)
                # We should only have mask around the target bodypart.
                # Maybe


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