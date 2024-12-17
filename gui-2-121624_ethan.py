import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time
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
user_name = ""

def get_dominant_color_hsv(image, k=2):
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
    # print(f'color1 is {lower_skin}')
    # print(f'color1 is {upper_skin}')
    # lower_skin = np.array([(dominant_color[0] - 20), 20, 50], dtype=np.uint8)
    # upper_skin = np.array([(dominant_color[0] + 20), 255, 255], dtype=np.uint8)

    # return dominant_color
    return lower_skin, upper_skin


def convert_grayscale_to_rgb_mask(grayscale_mask):
    # Create an RGB image by stacking the grayscale mask 3 times (initially, R=G=B)
    rgb_mask = cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR)

    # Create a binary mask where grayscale values are 255 (foreground)
    red_mask = grayscale_mask == 255

    # Set Red channel to 255 for the 255 pixels (leaving G and B as 0)
    rgb_mask[red_mask] = [0, 255, 0]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    return rgb_mask


def highlight_navy_blue(frame1, lower_skin, upper_skin):

    frame1 = cv2.flip(frame1, 1)

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)

    # Create a new frame where navy-blue pixels will be highlighted in red
    highlighted_frame = frame1.copy()

    # Replace navy-blue pixels with bright red in the original frame
    highlighted_frame[mask1 != 0] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    rgb_mask = convert_grayscale_to_rgb_mask(mask1)

    return highlighted_frame, rgb_mask


# Function definitions

# return frame as is, situation on starting up webcam
def function0(frame):
    """Applies a grayscale filter."""
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, text=f"Click 'Create Mask' when you are ready {user_name}", org=(25,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return cv2.GaussianBlur(frame, (15, 15), 0)

    # return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#create mask
def function1(frame):
    global rgb_mask
    print('First we will record your current position and create a mask - try and stay within this mask!')

    # Step 1: Use K-means clustering to identify the dominant background color
    def get_background_color(frame, k=2):
        """Identify the dominant background color using K-means clustering."""
        resized_frame = cv2.resize(frame, (100, 100))  # Resize for faster clustering
        pixels = resized_frame.reshape((-1, 3))  # Flatten the image to a 2D array
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(np.uint8)
        # Assume the background is the most frequent color cluster
        background_color = dominant_colors[np.argmax(np.unique(kmeans.labels_, return_counts=True)[1])]
        return background_color

    # Step 2: Detect Skin Regions in HSV Color Space
    def detect_skin_regions(frame):
        """Create a mask for skin regions based on HSV thresholds."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 48, 80], dtype=np.uint8)    # Lower HSV threshold for skin
        upper_skin = np.array([20, 255, 255], dtype=np.uint8) # Upper HSV threshold for skin
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)  # Threshold the HSV image
        return skin_mask

    # Step 3: Background Subtraction using K-means Background Color
    def background_subtraction(frame, bg_color, threshold=40):
        """Create a binary mask where the background is excluded."""
        diff = cv2.absdiff(frame, bg_color)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        return bg_mask

    # Step 4: Combine Skin Mask and Background Mask
    frame_flipped = cv2.flip(frame, 1)  # Flip for consistency
    bg_color = get_background_color(frame_flipped)  # Detect background color
    bg_mask = background_subtraction(frame_flipped, bg_color)  # Background mask
    skin_mask = detect_skin_regions(frame_flipped)  # Skin region mask

    # Combine skin mask and exclude background
    combined_mask = cv2.bitwise_and(skin_mask, bg_mask)

    # Step 5: Convert combined mask to RGB and overlay it
    rgb_mask = convert_grayscale_to_rgb_mask(combined_mask)  # Convert to RGB mask
    roi = frame_flipped[:, :]
    roi[np.where(rgb_mask)] = 0  # Apply the mask
    roi += rgb_mask.astype(np.uint8)

    # Step 6: Add instructions text
    cv2.putText(frame_flipped, text="Click 'Save Mask' if you are satisfied with the positioning",
                org=(25, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    return frame_flipped



#display the saved mask
def function2(frame):
    global rgb_mask
    frame = cv2.flip(frame, 1)
    # this is the section where we superimpose the initially captured mask
    roi = frame[:, :]
    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)
    print("saved")
    cv2.putText(frame, text="The mask has been saved, proceed to 'Determine Imaging Windows'", org=(25, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 0), thickness=2)

    return(frame)


#Base position calibration
def function3(frame):
    global rgb_mask, base_positions, base_width
    frame = cv2.flip(frame, 1)

    # Detect the ArUco marker before adding mask
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    roi = frame[:, :]
    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)


    # If marker detected, draw a bounding box around it
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        base_positions = corners
        # Extract the 2D array of coordinates
        coords = base_positions[0][0]

        # Calculate distances between consecutive corners (with wrapping to first corner)
        side_lengths = [np.linalg.norm(coords[i] - coords[(i + 1) % len(coords)]) for i in
                        range(len(coords))]

        # Calculate the average side length
        average_side_length = np.mean(side_lengths)

        print("Average side length:", average_side_length)
        base_width = average_side_length
        print("Base position saved.")
        cv2.putText(frame, text="Click 'Save Base Position' when you are ready", org=(25, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    else:
        # cv2.imshow('Base Marker Detection', frame)
        print("Please place the aruco marker as flat as possible on your sternum")
        cv2.putText(frame, text="Make sure the marker is visible", org=(25, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame


# save and draw base positions
def function4(frame):
    global rgb_mask, base_positions, base_width
    frame = cv2.flip(frame, 1)
    roi = frame[:, :]
    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)

    # draw original base square
    base_points = np.array(base_positions[0][0], np.int32)
    # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
    base_points = base_points.reshape((-1, 1, 2))
    # print(f"base corners are {base_points}")

    cv2.polylines(frame, [base_points], True, (255, 0, 0), 2)
    # """Applies an edge detection filter."""
    # return cv2.Canny(frame, 100, 200)
    cv2.putText(frame, text="The base position is saved, proceed to 'Determine Imaging Windows'", org=(25, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame


# Detect relative positions
def function5(frame):
    global positions_collected, total_positions, relative_positions, rgb_mask, base_positions

    frame = cv2.flip(frame, 1)

    # Detect the ArUco marker before adding mask
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    roi = frame[:, :]

    #dra base position
    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)

    # # draw original base square
    # base_points = np.array(base_positions[0][0], np.int32)
    # # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
    # base_points = base_points.reshape((-1, 1, 2))
    # # print(f"base corners are {base_points}")
    # cv2.polylines(frame, [base_points], True, (255, 0, 255), 2)

    # draw all saved  markers except the current one
    vertices_list = list(relative_positions.values())[:-1]
    if positions_collected > 0:
        for vertices_set in vertices_list:
            # Convert the set of vertices to a NumPy array of shape (n, 1, 2)
            vertices = np.array(list(vertices_set), dtype=np.int32).reshape(-1, 1, 2)

            # Draw the polygon using cv2.polylines
            cv2.polylines(frame, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

    if positions_collected <= total_positions:
        string1 = f"Position {positions_collected} of {total_positions} has been collected"
        cv2.putText(frame, text=string1, org=(25, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        # If marker detected, draw a bounding box around it
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            relative_positions[positions_collected] = corners
            print(f'All saved corners {relative_positions}')
            cv2.putText(frame, text="Click 'Save Imaging Window' when you get a good ultrasound image", org=(25, 150),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        else:
            cv2.putText(frame, text="Please display the marker clearly", org=(25, 150),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
            print("please display the marker clearly")
    else:
        print("all positions have been saved,, move on to overlap")
        cv2.putText(frame, text="All positions are saved, you can close the application", org=(25, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame


#save and draw relative positions
def function6(frame):
    global positions_collected, total_positions, relative_positions, rgb_mask, current_function_index, base_positions

    frame = cv2.flip(frame, 1)
    roi = frame[:, :]

    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)

    # # draw original base square
    # base_points = np.array(base_positions[0][0], np.int32)
    # # Reshape to match the (n, 1, 2) shape expected by cv2.polylines()
    # base_points = base_points.reshape((-1, 1, 2))
    # # print(f"base corners are {base_points}")
    #
    # cv2.polylines(frame, [base_points], True, (255, 0, 255), 2)

    if positions_collected > 0:
        for vertices_set in relative_positions.values():
            # Convert the set of vertices to a NumPy array of shape (n, 1, 2)
            vertices = np.array(list(vertices_set), dtype=np.int32).reshape(-1, 1, 2)

            # Draw the polygon using cv2.polylines
            cv2.polylines(frame, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

    # revert to function 5 to save the new position unless all positions have been saved
    if positions_collected < total_positions:
        current_function_index[0] = 5
    print(f"positions_collected {positions_collected}")
    cv2.putText(frame, text="All positions are saved, you can close the application or proceed to Realign Probe", org=(25, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame


#compare probe overlap
def function7(frame):
    global positions_collected, total_positions, relative_positions, rgb_mask, current_function_index

    frame = cv2.flip(frame, 1)

    # detect aruco markers before adding mask
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    roi = frame[:, :]

    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)

    aruco.drawDetectedMarkers(frame, corners, ids)

    # check overlap
    if ids is not None:
        print(f'corners of marker are {corners}')
        for i in range(total_positions):
            diff = np.abs(np.array(corners) - np.array(relative_positions[i]))
            if np.all(diff <= 10):
                # highlight in green if aligned
                cv2.polylines(frame, np.int32(relative_positions[i]), True, (0, 255, 0), 4)
                cv2.putText(frame, text="Aligned", org=(25, 150),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=3)
            else:
                # otherwise draw existing markers
                cv2.polylines(frame, np.int32(relative_positions[i]), True, (255, 0, 0), 2)

    cv2.putText(frame, text="Align the marker with the saved location", org=(25, 100),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame


# Main function
def main():
    global camera_matrix, dist_coeffs, base_positions, base_width, relative_positions, rgb_mask, total_positions, positions_collected, current_function_index
    global color_threshold_button
    total_positions = 2
    positions_collected = 0
    relative_positions = {}
    base_width = 0

    # Load camera calibration parameters
    # camera_matrix = np.load("camera_matrix_logi.npy")
    # dist_coeffs = np.load("dist_coeffs_logi.npy")

    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Demo 1: myFUS")

    # Initialize VideoCapture object
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    # UI Elements
    label_text = tk.StringVar()  # Variable to update the label
    label_text.set("Function: None")
    label = tk.Label(root, textvariable=label_text, font=("Helvetica", 16))
    label.pack()

    video_frame = tk.Label(root)
    video_frame.pack()

    # Function list and index
    functions = [function0, function1, function2, function3, function4, function5, function6, function7]
    function_names = ["Function 0 - Welcome, click to proceed when you are comfortable seated", "Function 1 - Start "
                                                                                                "color detection",
                      "Function 2, Save Mask", "Function 3, Base calib", "Function 4 , Save Base Position",
                      "Function 5, Determine imaging windows", "Function 6, Save imaging windows", "Function 7, Probe alignment"]
    current_function_index = [0] # Using a list to allow mutable behavior in the nested function


    def update_video():
        """Captures video frames and updates the display."""
        ret, frame = cap.read()
        if ret:
            # Apply the current function
            func = functions[current_function_index[0]]
            label_text.set(function_names[current_function_index[0]])
            # update_buttons()
            processed_frame = func(frame)

            # Convert processed frame for Tkinter display
            if len(processed_frame.shape) == 2:  # Grayscale
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            else:  # Colored frame
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(processed_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_frame.imgtk = imgtk
            video_frame.configure(image=imgtk)

        # Update the frame in the Tkinter GUI
        video_frame.after(10, update_video)


    def color_threshold():
        """Switch to the next function."""
        current_function_index[0] = 1

    color_threshold_button = tk.Button(root, text="Create Mask", wraplength=50, bg="lightblue", fg ="black", command=color_threshold, font=("Helvetica", 12))
    color_threshold_button.place(relx=0.1, rely=0.8, relheight=0.1, relwidth=0.2)

    def save_mask():
        current_function_index[0] = 2

    save_mask_button = tk.Button(root, text="Save Mask", bg="green", fg="black", command=save_mask, font=("Helvetica", 12))
    save_mask_button.place(relx=0.1, rely=0.9, relheight=0.1, relwidth=0.2)

    # def base_calibration():
    #     current_function_index[0] = 3
    #
    # base_calib_button = tk.Button(root, text="Start Base Calibration", bg="lightblue", fg ="black", command=base_calibration, font=("Helvetica", 12))
    # base_calib_button.place(relx=0.3, rely=0.8, relheight=0.1, relwidth=0.2)
    #
    # def save_base():
    #     current_function_index[0] = 4
    #
    # save_base_button = tk.Button(root, text="Save Base Position", command=save_base, bg="lightblue", fg ="black", font=("Helvetica", 12))
    # save_base_button.place(relx=0.3, rely=0.9, relheight=0.1, relwidth=0.2)

    def relative_calibration():
        global positions_collected
        current_function_index[0] = 5

    relative_calib_button = tk.Button(root, text="Determine Imaging Window", bg="lightblue", fg ="black",  command=relative_calibration,
                                  font=("Helvetica", 12))
    relative_calib_button.place(relx=0.3, rely=0.8, relheight=0.1, relwidth=0.2)

    def save_relative():
        global positions_collected
        current_function_index[0] = 6
        positions_collected += 1

    save_relative_button = tk.Button(root, text=f"Save Imaging Window", bg="lightblue", fg ="black", command=save_relative, font=("Helvetica", 12))
    save_relative_button.place(relx=0.3, rely=0.9, relheight=0.1, relwidth=0.2)

    def probe_overlap():
        current_function_index[0] = 7

    save_relative_button = tk.Button(root, text=f"Realign probe", command=probe_overlap, font=("Helvetica", 12))
    save_relative_button.place(relx=0.5, rely=0.9, relheight=0.1, relwidth=0.2)


    update_video()

    # Run Tkinter loop
    root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
    root.mainloop()


# Run the main function
if __name__ == "__main__":
    main()