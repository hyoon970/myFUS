import cv2
import tkinter as tk
from tkinter import Label
from tkinter import simpledialog
from PIL import Image, ImageTk
import time
import numpy as np
import cv2.aruco as aruco
from sklearn.cluster import KMeans
from collections import Counter
import os
import scipy.io

# Initialize the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_id = 23  # Change to the desired marker ID
marker_size = 0.0345  # Size of marker in meters (3.45 cm)
user_name = ""

# All button commands

## All button definitions
def color_threshold():
    """Switch to the next function."""
    current_function_index[0] = 1

def relative_calibration():
    global positions_collected
    current_function_index[0] = 3

def save_mask():
    current_function_index[0] = 2
    print("Saving Mask")

def save_relative():
    global positions_collected, current_function_index
    current_function_index[0] = 4
    positions_collected += 1

def display_mask():
    global current_function_index
    current_function_index[0] = 4

def probe_overlap():
    global current_function_index
    current_function_index[0] = 5

# all preliminary scripts for color thresholding for mask generation
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
    global rgb_mask
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

# all functions start here
def function0(frame):
    global instruction, user_name, mask_collected
    mask_collected = 0
    """Applies a grayscale filter."""
    frame = cv2.flip(frame, 1)
    # cv2.putText(frame, text=f"Click 'Mask Capture' when you are ready {user_name}", org=(150, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.1, color=(0, 0, 0), thickness=2)
    instruction = f"Click 'Mask Capture' when you are ready"
    return cv2.GaussianBlur(frame, (15, 15), 0)

#create mask
def function1(frame):
    global rgb_mask, instruction, mask_collected, mask_overlap
    global instruction, user_name, mask_collected
    mask_collected = 0
    print('First we will record your current position and create a mask - try and stay within this mask!')
    # mask is acquired here
    lower_skin, upper_skin = get_dominant_color_hsv(frame, k=2)
    highlighted_frame, rgb_mask = highlight_navy_blue(frame, lower_skin, upper_skin)
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, text="Click 'Save Mask' if you are satisfied with the positioning", org=(150, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    # this is the section where we superimpose the initially captured mask
    roi = frame[:, :]

    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)
    return frame

#display the saved mask
def function2(frame):
    global rgb_mask, total_saved, positions_collected, aruco_flag, mask_collected, mask_overlap
    frame = cv2.flip(frame, 1)
    modified_frame = frame.copy()

    # Dynamically calculate the transparent black overlay
    lower_skin, upper_skin = get_dominant_color_hsv(frame, k=2)
    dynamic_mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_skin, upper_skin)

    # Apply the transparent green overlay
    alpha_green = 0.5  # Transparency level
    dynamic_mask_indices = dynamic_mask != 0
    modified_frame[dynamic_mask_indices] = (
        (1 - alpha_green) * modified_frame[dynamic_mask_indices]
        + alpha_green * np.array([0, 255, 0], dtype=np.uint8)
    )

    # Calculate overlap percentage between the current mask and saved mask
    grayscale_saved_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
    saved_mask_indices = grayscale_saved_mask != 0

    overlap = np.logical_and(dynamic_mask_indices, saved_mask_indices).sum()
    total_saved = saved_mask_indices.sum()

    match_percentage = (overlap / total_saved * 100) if total_saved > 0 else 0
    mask_collected = 1
    mask_overlap = ceil(match_percentage)

    # Create a transparent white overlay for the saved mask
    overlay = modified_frame.copy()
    contours, _ = cv2.findContours(grayscale_saved_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the larger white outline on the overlay
    for contour in contours:
        cv2.polylines(overlay, [contour], isClosed=True, color=(255, 255, 255), thickness=10)  # White (BGR)

    # Blend the white overlay with the frame
    alpha_white = 0.7  # Transparency level for white outline
    cv2.addWeighted(overlay, alpha_white, modified_frame, 1 - alpha_white, 0, modified_frame)

    # Draw the smaller pink outline on top of everything
    for contour in contours:
        cv2.polylines(modified_frame, [contour], isClosed=True, color=(203, 192, 255), thickness=2)  # Pink (BGR)

    # Display "Mask" Information
    # if total_saved > 0:  # Display only after a mask is saved
    #       cv2.putText(modified_frame, "Mask", org=(80, 300), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
    #                 fontScale=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    #
    #     cv2.putText(modified_frame, f"Mask Collected: Yes",

    #                 org=(80, 350), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
    #                 color=(255, 255, 400), thickness=2, lineType=cv2.LINE_AA)
    #
    #     cv2.putText(modified_frame, f"Mask Overlap: {match_percentage:.1f}%",
    #                 org=(80, 385), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
    #                 color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    #
    #     # Add instruction text in color if necessary
    #     instruction_text = "Move into position" if match_percentage < 90 else "Good position"
    #     text_color = (80, 100, 255)  # Light pink color
    #     cv2.putText(modified_frame, instruction_text, org=(80, 420), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
    #                 fontScale=0.5, color=text_color, thickness=2, lineType=cv2.LINE_AA)

def function3(frame2):
    global positions_collected, total_positions, relative_positions, rgb_mask, current_function_index, total_saved, aruco_flag
    global probe_coordinates, mask_collected, mask_overlap  # Store probe coordinates

    aruco_flag = 1
    # saved mask needs to be displayed for all functions henceforth
    frame = function2(frame2)

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
        cv2.putText(frame, text=string1, org=(100, 100),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        # continously detect new markers
        # while True:
        # detect aruco markers
        corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
        # If marker detected, draw a bounding box around it
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            relative_positions[positions_collected] = corners
            print(f'All saved corners {relative_positions}')
            cv2.putText(frame, text="Click 'Save Imaging Window' when you get a good ultrasound image",
                        org=(100, 200),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.75, color=(0, 0, 0), thickness=2)

            # Save the detected corners (x1, y1) ... (x4, y4)
            for marker in corners:
                # Convert marker coordinates to a list of (x, y) tuples
                corner_list = [(int(pt[0]), int(pt[1])) for pt in marker[0]]
                probe_coordinates.append(corner_list)  # Save to global list
                print(f"Saved probe coordinates: {corner_list}")

        else:
            cv2.putText(frame, text="Please display the marker clearly", org=(100, 200),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.75, color=(0, 0, 0), thickness=2)
            print("please display the marker clearly")
    else:
        print("all positions have been saved,, move on to overlap")
        cv2.putText(frame, text="All positions are saved, you can close the application or proceed to Realign Probe",
                    org=(100, 100),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame

# save and draw relative positions
def function4(frame2):
    global positions_collected, total_positions, relative_positions, rgb_mask, current_function_index

    frame = function2(frame2)

    # Here the relative positions display logic begins
    if positions_collected > 0:
        for vertices_set in relative_positions.values():
            # Convert the set of vertices to a NumPy array of shape (n, 1, 2)
            vertices = np.array(list(vertices_set), dtype=np.int32).reshape(-1, 1, 2)

            # Draw the polygon using cv2.polylines
            cv2.polylines(frame, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

    # revert to function 5 to save the new position unless all positions have been saved
    if positions_collected < total_positions:
        current_function_index[0] = 3
        print(f"positions_collected {positions_collected}")
        return frame
    else:
        cv2.putText(frame, text="All positions are saved, you can close the application or proceed to Realign Probe",
                    org=(100, 100),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.75, color=(0, 0, 0), thickness=2)

    return frame

# compare probe overlap
def function5(frame2):
    global positions_collected, total_positions, relative_positions, rgb_mask, current_function_index

    frame = function2(frame2)

    # Here the probe overlap comparison begins

    # declare tolerance for corner mismatch
    tolerance = 10

    for i in range(total_positions):
        cv2.polylines(frame, np.int32(relative_positions[i]), True, (255, 0, 0), 2)

    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    aruco.drawDetectedMarkers(frame, corners, ids)
    overlap_id = 0

    # check overlap
    if ids is not None:
        # print(f'corners of marker are {corners}')
        # print(f'corners of saved position 1 are {relative_positions[0]}')
        # print(f'corners of saved position 2 are {relative_positions[1]}')
        for i in range(total_positions):
            diff = np.abs(np.array(corners) - np.array(relative_positions[i]))
            print(f'difference array is {diff}')
            if np.all(diff <= tolerance):
                overlap_id = i
                print(overlap_id)
                cv2.putText(frame, text=f"Overlap detected w position {overlap_id}", org=(100, 200),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
                cv2.polylines(frame, np.int32(relative_positions[overlap_id]), True, (0, 255, 0), 4)
            else:
                cv2.putText(frame, text=f"Align with blue highlighted square", org=(100, 100),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
                cv2.polylines(frame, np.int32(relative_positions[overlap_id]), True, (255, 0, 0), 4)
        # Draw the filled polygon
        # fill_color = (0, 255, 0)  # Fill color in BGR (Green in this case)
        # cv2.fillPoly(frame, [np.array(relative_corners[overlap_id])], fill_color)
    return frame

def instructionstr():
    global instruction
    return instruction

def maskdata():
    global mask_collected, mask_overlap
    mask_data = f"Mask Collected (0/1): {mask_collected} \n Mask Overlap: {mask_overlap}"
    return mask_data

# Placeholder for `current_frame`
current_frame = np.zeros((1280, 720, 3), dtype=np.uint8)  # Dummy frame for initialization

def setup_gui():
    global root, current_frame, video_frame, instruction, maskstats, mask_collected, mask_overlap
    global color_threshold_button, save_mask_button, relative_calib_button, save_relative_button, display_mask_button, comp_probe_overlap_button

    root = tk.Tk()
    root.title("Demo 1: myFUS")


    # Set window size (adjust width to fit panels)
    window_width = 1600  # Increased width to fit left panel
    window_height = 1200  # Increased height to fit top panel
    root.geometry(f"{window_width}x{window_height}")

    # Create the **Top Panel** for messages
    top_panel = tk.Frame(root, height=240, bg="lightgray")  # Light gray background
    top_panel.pack(side="top", fill="x")  # Expands horizontally

    instruction_label = tk.StringVar()
    instruction_label.set(instruction)
    message_label = tk.Label(top_panel, textvariable=instruction_label, font=("Helvetica", 14), bg="lightgray")
    message_label.pack(pady=10)  # Add spacing inside the panel

    # Create the **Left Panel** for buttons
    left_panel = tk.Frame(root, width=374, bg="lightgray")  # Fixed width
    left_panel.pack(side="left", fill="y")  # Expands vertically

    # Create the **Mask Overlap Stats**
    mask_panel = tk.Frame(left_panel, width=240, height=200, bg="lightblue")  # Fixed width
    mask_panel.pack(side="bottom", padx = 20, pady=40)  # Fit in the centre of Left Panel at 40 px from bottom

    # Add a label for mask stats inside the panel
    # Initialize mask stats
    maskstats = tk.StringVar()
    maskstats.set(maskdata())  # Pulls the current instruction to display
    maskstats_label = tk.Label(mask_panel, textvariable=maskstats, font=("Helvetica", 14), bg="white")
    maskstats_label.pack(padx=20, pady=20)  # Add spacing inside the panel

    # Created video-frame widget to display current frame within the GUI window
    video_frame = tk.Label(root)
    video_frame.pack(expand=True, fill="both")  # Expands to remaining space

    # Defining all buttons:

    color_threshold_button = tk.Button(left_panel, text="Mask Capture", wraplength=50, bg="lightblue", fg="black",
                                       command=save_mask, font=("Helvetica", 12))
    color_threshold_button.pack(padx=20, pady=20, fill="x")


    save_mask_button = tk.Button(left_panel, text="Save Mask", bg="green", fg="black", command=save_mask,
                                 font=("Helvetica", 12))
    save_mask_button.pack(padx=20, pady=20, fill="x")


    relative_calib_button = tk.Button(left_panel, text="Find Imaging Windows", bg="lightblue", fg="black",
                                      command=relative_calibration,
                                      font=("Helvetica", 12))
    relative_calib_button.pack(padx=20, pady=20, fill="x")


    save_relative_button = tk.Button(left_panel, text=f"Save Imaging Window", bg="lightblue", fg="black",
                                     command=save_mask, font=("Helvetica", 12))
    save_relative_button.pack(padx=20, pady=20, fill="x")


    display_mask_button = tk.Button(left_panel, text=f"Display Mask", command=save_mask, font=("Helvetica", 12))
    display_mask_button.pack(padx=20, pady=20, fill="x")


    comp_probe_overlap_button = tk.Button(root, text=f"Realign probe", command=save_mask, font=("Helvetica", 12))
    # comp_probe_overlap_button.place(relx=0.8, rely=0.87, relheight=0.1, relwidth=0.115)
    comp_probe_overlap_button.pack(padx=20, pady=20, fill="x")


    return root

def main():
    global current_frame
    global maskstats, video_frame, instruction
    global mask_collected, mask_overlap
    global camera_matrix, dist_coeffs, base_positions, base_width, relative_positions, rgb_mask, total_positions, positions_collected, current_function_index, aruco_flag
    global color_threshold_button
    global user_name, user_folder
    global color_threshold_button, save_mask_button, relative_calib_button, save_relative_button, display_mask_button, comp_probe_overlap_button
    aruco_flag = 0
    total_positions = 2
    positions_collected = 0
    relative_positions = {}
    user_name = "aastha"
    instruction = "Welcome! Instructions will appear here."
    mask_collected = 0
    mask_overlap = 0

    functions = [function0, function1, function2, function3, function4, function5]
    current_function_index = [0]  # Using a list to allow mutable behavior in the nested function

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    def update_video():
        global current_frame, maskstats, root
        ret, raw_frame = cap.read()
        if ret:
            # calling all functions that need to be refreshed every 10 ms
            func = functions[current_function_index[0]]
            # label_text_var.set(function_names[current_function_index[0]])
            instructionstr()
            # Setup the GUI
            update_buttons()  # Update buttons dynamically
            frame = func(raw_frame)


            # Scale the frame to 3/4th of its original size
            height, width = frame.shape[:2]  # Get the original dimensions
            scaled_width = int(width * 1)  # 3/4th of the width
            scaled_height = int(height * 1)  # 3/4th of the height
            scaled_frame = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

            # Update current_frame with the captured frame
            current_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)

            # Convert current_frame to an ImageTk.PhotoImage and update the GUI
            img = Image.fromarray(current_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_frame.imgtk = imgtk
            video_frame.configure(image=imgtk)

        # Schedule the next update
        video_frame.after(10, update_video)

    def update_buttons():

        """Enable or disable buttons based on the current function."""
        # color_threshold_button.config(state=tk.DISABLED)
        save_mask_button.config(state=tk.DISABLED)
        relative_calib_button.config(state=tk.DISABLED)
        save_relative_button.config(state=tk.DISABLED)
        display_mask_button.config(state=tk.DISABLED)
        comp_probe_overlap_button.config(state=tk.DISABLED)
        aruco_flag = 0
        #
        if current_function_index[0] == 0:  # Initial state
            color_threshold_button.config(state=tk.NORMAL)
        elif current_function_index[0] == 1:  # Create Mask
            save_mask_button.config(state=tk.NORMAL)
        elif current_function_index[0] == 2:  # Save Mask
            relative_calib_button.config(state=tk.NORMAL)
            aruco_flag = 1
        elif current_function_index[0] == 3:  # Determine Imaging Window
            save_relative_button.config(state=tk.NORMAL)
        elif current_function_index[0] == 4:  # Save Imaging Window
            if positions_collected < total_positions:
                save_relative_button.config(state=tk.NORMAL)
            else:
                display_mask_button.config(state=tk.NORMAL)
                comp_probe_overlap_button.config(state=tk.NORMAL)


    print(current_function_index)
    # Setup the GUI
    root = setup_gui()

    # Start the video update loop
    update_video()

    # Run the Tkinter main loop
    root.mainloop()

    # Release the video capture on close
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
