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
    cv2.putText(frame, text=f"Click 'Mask Capture' when you are ready {user_name}", org=(150, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.1, color=(0, 0, 0), thickness=2)
    return cv2.GaussianBlur(frame, (15, 15), 0)

    # return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#create mask
def function1(frame):
    global rgb_mask
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
    global rgb_mask, total_saved, positions_collected, aruco_flag

    ##########################
    ### Thing that I Added ###
    global flag_overlap
    ##########################


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


    ###########################
    ### Flag of the overlap ###
    ###########################
    flag_overlap = 0
    if match_percentage >= 90:
        flag_overlap = 1
    elif match_percentage < 90:
        flag_overlap = 0
    ###########################



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
    if total_saved > 0:  # Display only after a mask is saved
        cv2.putText(modified_frame, "Mask", org=(80, 300), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        cv2.putText(modified_frame, f"Mask Collected: Yes",
                    org=(80, 350), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                    color=(255, 255, 400), thickness=2, lineType=cv2.LINE_AA)

        cv2.putText(modified_frame, f"Mask Overlap: {match_percentage:.1f}%",
                    org=(80, 385), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5,
                    color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Add instruction text in color if necessary
        instruction_text = "Move into position" if match_percentage < 90 else "Good position"
        text_color = (80, 100, 255)  # Light pink color
        cv2.putText(modified_frame, instruction_text, org=(80, 420), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=0.5, color=text_color, thickness=2, lineType=cv2.LINE_AA)

    # Display the match percentage on the frame
    # percentage_text = f"Match: {match_percentage:.2f}%"
    # cv2.putText(modified_frame, percentage_text, org=(250, 200), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
    #             fontScale=1, color=(0, 255, 0), thickness=2)

    # Add instruction text
    if aruco_flag != 1:
        cv2.putText(modified_frame, text="The mask has been saved, proceed to 'Find Imaging Windows'",
                org=(150, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0),
                thickness=2)

    return modified_frame

# Detect aruco marker positions
def function3(frame2):
    global positions_collected, total_positions, relative_positions, rgb_mask, current_function_index, total_saved, aruco_flag

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
        else:
            cv2.putText(frame, text="Please display the marker clearly", org=(100, 200),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.75, color=(0, 0, 0), thickness=2)
            print("please display the marker clearly")
    else:
        print("all positions have been saved,, move on to overlap")
        cv2.putText(frame, text="All positions are saved, you can close the application or proceed to Realign Probe", org=(100, 100),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    return frame

#save and draw relative positions
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
        cv2.putText(frame, text="All positions are saved, you can close the application or proceed to Realign Probe", org=(100, 100),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.75, color=(0, 0, 0), thickness=2)

    return frame


#compare probe overlap
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
        #fill_color = (0, 255, 0)  # Fill color in BGR (Green in this case)
        # cv2.fillPoly(frame, [np.array(relative_corners[overlap_id])], fill_color)
    return frame

# Main function
def main():
    global camera_matrix, dist_coeffs, base_positions, base_width, relative_positions, rgb_mask, total_positions, positions_collected, current_function_index, aruco_flag
    global color_threshold_button
    aruco_flag = 0
    total_positions = 2
    positions_collected = 0
    relative_positions = {}
    base_width = 0

    # Load camera calibration parameters
    camera_matrix = np.load("camera_matrix_logi.npy")
    dist_coeffs = np.load("dist_coeffs_logi.npy")

    # Make logo black

    # Load the logo image
    logo = cv2.imread("decoders_logo.png", cv2.IMREAD_UNCHANGED)

    # Resize the logo if needed
    logo_height, logo_width = 75, 75  # Desired dimensions
    logo = cv2.resize(logo, (logo_width, logo_height), interpolation=cv2.INTER_AREA)

    # Split the logo into color and alpha channels
    if logo.shape[2] == 4:  # Check if the logo has an alpha channel
        logo_bgr = logo[:, :, :3]
        logo_alpha = logo[:, :, 3] / 255.0  # Normalize alpha channel to range [0, 1]
    else:
        logo_bgr = logo
        logo_alpha = np.ones(logo_bgr.shape[:2], dtype=np.float32)  # No transparency


    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Demo 1: myFUS")

    # Initialize VideoCapture object
    cap = cv2.VideoCapture(0)  # 0 for the default webcam

    # UI Elements
    label_text_var = tk.StringVar()  # Variable to update the label
    label_text_var.set("Function: None")
    dynamic_label = tk.Label(root, textvariable=label_text_var, font=("Helvetica", 16))
    dynamic_label.place(relx=0.2, rely=0.2, relheight=0.1, relwidth=0.6)
    # dynamic_label.pack()

    # Add a label above the column of buttons
    calibration_label = tk.Label(root, text="Start-Up\nCalibration", bg="black", fg="white",
                                font=("Helvetica", 16, "bold"), justify="center")
    calibration_label.place(relx=.7, rely=.1, relheight=.1, relwidth=.115)

    video_frame = tk.Label(root)
    video_frame.pack()

    # Function list and index
    functions = [function0, function1, function2, function3, function4, function5]
    # functions = [function00 (ethan create user profile to save mask and windows), function0, function1, function2, function3, function4, function5]
    function_names = ["Function 0 - Welcome, click 'Save Mask' to proceed when you are comfortably seated", "Function "
                                                                                                            "1 - Start"
                                                                                                "color detection",
                      "Function 2, Save Mask",
                      "Function 3, Determine imaging windows", "Function 4 - Save imaging windows","Function 5, Probe "
                                                                                                   "alignment"]
    current_function_index = [0] # Using a list to allow mutable behavior in the nested function


    def update_buttons():

        
        """Enable or disable buttons based on the current function."""
        color_threshold_button.config(state=tk.DISABLED)
        save_mask_button.config(state=tk.DISABLED)
        relative_calib_button.config(state=tk.DISABLED)
        save_relative_button.config(state=tk.DISABLED)
        display_mask_button.config(state=tk.DISABLED)
        comp_probe_overlap_button.config(state=tk.DISABLED)
        aruco_flag = 0

        if current_function_index[0] == 0:  # Initial state
            color_threshold_button.config(state=tk.NORMAL)
        elif current_function_index[0] == 1:  # Create Mask
            save_mask_button.config(state=tk.NORMAL)
        elif current_function_index[0] == 2:  # Save Mask
            relative_calib_button.config(state=tk.NORMAL)
            aruco_flag = 1

        ############################
        ### Thing that I changed ###
        elif current_function_index[0] == 3:  # Determine Imaging Window
            if flag_overlap == 1:
                save_relative_button.config(state=tk.NORMAL)    ## Save Image Window Button
            elif flag_overlap == 0:
                save_relative_button.config(state=tk.DISABLED)
        ############################

        elif current_function_index[0] == 4:  # Save Imaging Window
            if positions_collected < total_positions:
                save_relative_button.config(state=tk.NORMAL)
            else:
                display_mask_button.config(state=tk.NORMAL)
                comp_probe_overlap_button.config(state=tk.NORMAL)


    # Modify the update_video function to include update_buttons
    def update_video():
        """Captures video frames and updates the display."""
        ret, frame = cap.read()
        if ret:
            func = functions[current_function_index[0]]
            label_text_var.set(function_names[current_function_index[0]])
            update_buttons()  # Update buttons dynamically
            processed_frame = func(frame)

            # Overlay the logo on the top-left corner of the video feed
            # Adjust the position of the logo
            offset_x = 20  # Adjust this value to move the logo to the right
            offset_y = 20  # Adjust this value to move the logo down
            y1, y2 = offset_y, offset_y + logo_height
            x1, x2 = offset_x, offset_x + logo_width

            # Extract the region of interest (ROI) from the frame
            roi = processed_frame[y1:y2, x1:x2]

            # Blend the logo with the ROI
            for c in range(3):  # Loop over B, G, R channels
                roi[:, :, c] = roi[:, :, c] * (1 - logo_alpha) + logo_bgr[:, :, c] * logo_alpha

            # Place the blended ROI back into the frame
            processed_frame[y1:y2, x1:x2] = roi

            if len(processed_frame.shape) == 2:  # Grayscale
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
            else:  # Colored frame
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(processed_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            video_frame.imgtk = imgtk
            video_frame.configure(image=imgtk)

        video_frame.after(10, update_video)



    def color_threshold():
        """Switch to the next function."""
        current_function_index[0] = 1

    color_threshold_button = tk.Button(root, text="Mask Capture", wraplength=50, bg="lightblue", fg ="black", command=color_threshold, font=("Helvetica", 12))
    color_threshold_button.place(relx=0.8, rely=0.15, relheight=0.1, relwidth=0.115)

    def save_mask():
        global aruco_flag
        current_function_index[0] = 2

    save_mask_button = tk.Button(root, text="Save Mask", bg="green", fg="black", command=save_mask, font=("Helvetica", 12))
    save_mask_button.place(relx=0.8, rely=0.27, relheight=0.1, relwidth=0.115)

    def relative_calibration():
        global positions_collected
        current_function_index[0] = 3

    relative_calib_button = tk.Button(root, text="Find Imaging Windows", bg="lightblue", fg ="black",  command=relative_calibration,
                                  font=("Helvetica", 12))
    relative_calib_button.place(relx=0.8, rely=0.39, relheight=0.1, relwidth=0.115)

    def save_relative():
        global positions_collected
        current_function_index[0] = 4
        positions_collected += 1

    save_relative_button = tk.Button(root, text=f"Save Imaging Window", bg="lightblue", fg ="black", command=save_relative, font=("Helvetica", 12))
    save_relative_button.place(relx=0.8, rely=0.51, relheight=0.1, relwidth=0.115)

    def display_mask():
        current_function_index[0] = 4

    display_mask_button = tk.Button(root, text=f"Display Mask", command=display_mask, font=("Helvetica", 12))
    display_mask_button.place(relx=0.8, rely=0.75, relheight=0.1, relwidth=0.115)


    def probe_overlap():
        current_function_index[0] = 5

    comp_probe_overlap_button = tk.Button(root, text=f"Realign probe", command=probe_overlap, font=("Helvetica", 12))
    comp_probe_overlap_button.place(relx=0.8, rely=0.87, relheight=0.1, relwidth=0.115)


    update_video()

    # Run Tkinter loop
    root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
    root.mainloop()


# Run the main function
if __name__ == "__main__":
    main()