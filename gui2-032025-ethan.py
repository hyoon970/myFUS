import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import cv2.aruco as aruco
from sklearn.cluster import KMeans
from collections import Counter
from tkinter import simpledialog
import os
import csv
from tkinter import PhotoImage

class MyFUSApp:
    def __init__(self):
        self.setup_variables()
        self.setup_gui()
        self.setup_camera()

    def setup_variables(self):
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco.DetectorParameters())
        self.marker_id = 23
        self.marker_size = 0.0345
        self.username = ""
        self.userfolder = ""
        self.masksfolder = ""
        self.probecoordsfolder = ""
        self.startup_flag = 1
        self.flip_flag = 0
        self.positions_collected = 0
        self.total_positions = 1
        self.relative_positions = {}
        self.current_function_index = 0
        self.mask_collected = 0
        self.mask_overlap = 0
        self.instruction = "Welcome! Instructions will appear here."
        self.rgb_mask = None
        self.probe_coordinates = []

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Demo 1: myFUS")
        self.root.geometry("1600x1200")
        self.root.configure(bg="black")  # Set main window background to black
        if self.startup_flag == 1:
            self.get_username()
        self.setup_panels()
        self.setup_buttons()

    def setup_panels(self):
        self.top_panel = tk.Frame(self.root, height=240, bg="black")
        self.top_panel.pack(side="top", fill="x")

        ### Thing that I added ###
        # Load the logo image
        original_logo_image = PhotoImage(file='decoders_logo.png')  # Replace with the actual path if needed
        self.logo_image = original_logo_image.subsample(2, 2)  # Resizing the image to half of its original size
        self.logo_label = tk.Label(self.top_panel, image=self.logo_image, bg="black")
        self.logo_label.pack(side="left", anchor="nw", pady=10, padx=10)
        ###########################

        self.instruction_var = tk.StringVar()
        self.instruction_var.set(self.instruction)
        self.instruction_label = tk.Label(self.top_panel, textvariable=self.instruction_var, font=("Helvetica", 14),
                                          bg="black", fg="white", wraplength=1500)
        self.instruction_label.pack(pady=10)

        self.left_panel = tk.Frame(self.root, width=374, bg="black")
        self.left_panel.pack(side="left", fill="y")

        self.mask_panel = tk.Frame(self.left_panel, width=240, height=200, bg="black")
        self.mask_panel.pack(side="bottom", padx=20, pady=40)

        self.maskstats = tk.StringVar()
        self.maskstats.set(self.get_mask_data())
        tk.Label(self.mask_panel, textvariable=self.maskstats, font=("Helvetica", 14),
                 bg="black", fg="white").pack(padx=20, pady=20)

        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(expand=True, fill="both")

    def setup_buttons(self):
        button_configs = [
            ("Mask Capture", self.color_threshold),
            ("Save Mask", self.save_mask),
            ("Find Imaging Windows", self.relative_calibration),
            ("Save Imaging Window", self.save_relative),
            ("Display Mask", self.display_mask),
            ("Realign probe", self.probe_overlap)
        ]

        self.buttons = []
        for text, command in button_configs:
            button = tk.Button(self.left_panel, text=text, command=command, font=("Helvetica", 12),
                               bg="black", fg="black")
            button.pack(padx=20, pady=20, fill="x")
            self.buttons.append(button)

        self.update_buttons()

    def setup_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.process_frame(frame)
            self.display_frame(frame)
        self.root.after(10, self.update_video)

    def process_frame(self, frame):
        functions = [self.function0, self.function1, self.function2, self.function3, self.function4, self.function5]
        return functions[self.current_function_index](frame)

    def display_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1280, 720))
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_frame.imgtk = photo
        self.video_frame.configure(image=photo)

    def update_buttons(self):
        for button in self.buttons:
            button.config(state=tk.DISABLED)

        if self.current_function_index == 0:
            self.buttons[0].config(state=tk.NORMAL)  # Enable "Mask Capture"
        elif self.current_function_index == 1:
            self.buttons[1].config(state=tk.NORMAL)  # Enable "Save Mask"
        elif self.current_function_index == 2:
            self.buttons[2].config(state=tk.NORMAL)  # Enable "Find Imaging Windows"
        elif self.current_function_index == 3:
            self.buttons[3].config(state=tk.NORMAL)  # Enable "Save Imaging Window"
        elif self.current_function_index == 4:
            if self.positions_collected < self.total_positions:
                self.buttons[3].config(state=tk.NORMAL)  # Enable "Save Imaging Window"
            else:
                self.buttons[4].config(state=tk.NORMAL)  # Enable "Display Mask"
                self.buttons[5].config(state=tk.NORMAL)  # Enable "Realign probe"

    def update_instruction(self, text):
        self.instruction_var.set(text)

    def get_mask_data(self):
        return f"Mask Collected (0/1): {self.mask_collected} \nMask Overlap: {self.mask_overlap}"

    def update_mask_stats(self):
        self.maskstats.set(self.get_mask_data())

    # Button command methods
    def color_threshold(self):
        self.current_function_index = 1
        self.update_buttons()

    def save_mask(self):
        self.current_function_index = 2
        self.update_buttons()

    def relative_calibration(self):
        self.current_function_index = 3
        self.update_buttons()

    def save_relative(self):
        self.current_function_index = 4
        self.positions_collected += 1
        self.update_buttons()

    def display_mask(self):
        self.current_function_index = 4
        self.update_buttons()

    def probe_overlap(self):
        self.current_function_index = 5
        self.update_buttons()

    def get_username(self):
        print(f"Current working directory: {os.getcwd()}")
        self.username = simpledialog.askstring("User Login", "Enter your username:")

        if not self.username:
            print("No username entered. Exiting.")
            exit()  # Exit if no username is provided

        # Define the user's folder path
        self.userfolder = os.path.join("users", self.username)

        # Create the folder only if it doesn’t already exist
        if not os.path.exists(self.userfolder):
            os.makedirs(self.userfolder)
            print(f"New user folder created: {self.userfolder}")
        else:
            print(f"Existing user folder found: {self.userfolder}")
        self.startup_flag = 0

    # Frame processing methods
    def function0(self, frame):
        self.mask_collected = 0
        self.update_mask_stats()
        frame = cv2.flip(frame, 1)
        self.update_instruction("Click 'Mask Capture' when you are ready")
        return cv2.GaussianBlur(frame, (15, 15), 0)

    def function1(self, frame):
        self.mask_collected = 0
        self.update_mask_stats()
        lower_skin, upper_skin = self.get_dominant_color_hsv(frame, k=2)
        highlighted_frame, self.rgb_mask = self.highlight_navy_blue(frame, lower_skin, upper_skin)
        frame = cv2.flip(frame, 1)
        self.update_instruction("Click 'Save Mask' if you are satisfied with the positioning")
        roi = frame[:, :]
        roi[np.where(self.rgb_mask)] = 0
        roi += self.rgb_mask.astype(np.uint8)
        return frame

    def function2(self, frame):
        frame = cv2.flip(frame, 1)
        modified_frame = frame.copy()
        lower_skin, upper_skin = self.get_dominant_color_hsv(frame, k=2)
        dynamic_mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower_skin, upper_skin)

        alpha_green = 0.5
        dynamic_mask_indices = dynamic_mask != 0
        modified_frame[dynamic_mask_indices] = (
                (1 - alpha_green) * modified_frame[dynamic_mask_indices]
                + alpha_green * np.array([0, 255, 0], dtype=np.uint8)
        )

        # Save mask to png in userfolder if it hasn't been saved
        if self.mask_collected == 0 and self.userfolder:
            gray_mask = cv2.cvtColor(self.rgb_mask, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_mask, 50, 255, cv2.THRESH_BINARY)
            self.masksfolder = os.path.join(self.userfolder, "masks")
            os.makedirs(self.masksfolder, exist_ok=True)
            mask_image_path = os.path.join(self.masksfolder, f"saved_mask_{self.username}.png")
            cv2.imwrite(mask_image_path, binary_mask)

        grayscale_saved_mask = cv2.cvtColor(self.rgb_mask, cv2.COLOR_BGR2GRAY)
        saved_mask_indices = grayscale_saved_mask != 0
        overlap = np.logical_and(dynamic_mask_indices, saved_mask_indices).sum()
        total_saved = saved_mask_indices.sum()
        match_percentage = (overlap / total_saved * 100) if total_saved > 0 else 0
        self.mask_collected = 1
        self.mask_overlap = int(match_percentage)
        self.update_mask_stats()

        overlay = modified_frame.copy()
        contours, _ = cv2.findContours(grayscale_saved_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.polylines(overlay, [contour], isClosed=True, color=(255, 255, 255), thickness=10)
        alpha_white = 0.7
        cv2.addWeighted(overlay, alpha_white, modified_frame, 1 - alpha_white, 0, modified_frame)
        # Changed outline color from pink to dark green (BGR: (0, 100, 0))
        for contour in contours:
            cv2.polylines(modified_frame, [contour], isClosed=True, color=(0, 100, 0), thickness=2)
        self.update_instruction(f"Mask overlap: {self.mask_overlap}%")
        return modified_frame

    def function3(self, frame):
        flipped_frame = cv2.flip(frame, 1)
        vertices_list = list(self.relative_positions.values())[:-1]
        if self.positions_collected > 0:
            for vertices_set in vertices_list:
                vertices = np.array(list(vertices_set), dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)

        if self.positions_collected <= self.total_positions:
            self.update_instruction(f"Position {self.positions_collected} of {self.total_positions} has been collected")
            corners, ids, rejected = aruco.detectMarkers(flipped_frame, self.aruco_dict,
                                                         parameters=self.detector.getDetectorParameters())

            if ids is not None:
                aruco.drawDetectedMarkers(flipped_frame, corners, ids)
                self.relative_positions[self.positions_collected] = corners[0][0]
                self.update_instruction("Click 'Save Imaging Window' when you get a good ultrasound image")
                for marker in corners:
                    corner_list = [(int(pt[0]), int(pt[1])) for pt in marker[0]]
                    self.probe_coordinates.append(corner_list)
            else:
                self.update_instruction("Please display the marker clearly")
        else:
            self.update_instruction(
                "All positions are saved, you can close the application or proceed to Realign Probe")
        flipped_frame = cv2.flip(flipped_frame, 1)
        frame = self.function2(flipped_frame)
        return frame

    def function4(self, frame):
        frame = self.function2(frame)
        if self.positions_collected > 0:
            for vertices_set in self.relative_positions.values():
                vertices = np.array(vertices_set, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [vertices], isClosed=True, color=(255, 0, 0), thickness=2)
        if self.positions_collected < self.total_positions:
            self.current_function_index = 3
            self.update_buttons()
            return frame
        else:
            print(self.relative_positions)
            try:
                all_positions = []
                for key, value in self.relative_positions.items():
                    positions_array = self.relative_positions[key]
                    print(positions_array)
                    for row in positions_array:
                        all_positions.append(row.flatten())

                self.probecoordsfolder = os.path.join(self.userfolder, "probecoords")
                os.makedirs(self.probecoordsfolder, exist_ok=True)
                output_file = os.path.join(self.probecoordsfolder, f"saved_probecoords_{self.username}.csv")
                with open(output_file, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(all_positions)
                print(f"Relative positions exported to {output_file}.")
            except Exception as e:
                print(f"An error occurred while exporting: {e}")

            self.update_instruction(
                "All positions are saved, you can close the application or proceed to Realign Probe")

        return frame

    def function5(self, frame):
        flipped_frame = cv2.flip(frame, 1)

        input_file = os.path.join(self.probecoordsfolder, f"saved_probecoords_{self.username}.csv")
        if os.path.exists(input_file):
            print(f"Reading data from: {input_file}")
            try:
                with open(input_file, mode="r") as file:
                    reader = csv.reader(file)
                    data = np.array(list(reader), dtype=float)
                    num_positions = len(data) // 4
                    print(f"number of saved quads {num_positions}")

                for i in range(0, len(data), 4):
                    coords = data[i:i + 4].astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [coords], isClosed=True, color=(0, 0, 255), thickness=2)

            except Exception as e:
                print(f"An error occurred while reading and drawing bounding boxes: {e}")
        else:
            print(f"Input file not found: {input_file}")

        tolerance = 10
        for i in range(self.total_positions):
            if i in self.relative_positions and len(self.relative_positions[i]) > 0:
                cv2.polylines(flipped_frame, np.int32([self.relative_positions[i]]), True, (255, 0, 0), 2)
        corners, ids, rejected = aruco.detectMarkers(flipped_frame, self.aruco_dict,
                                                     parameters=self.detector.getDetectorParameters())
        if corners:
            aruco.drawDetectedMarkers(flipped_frame, corners, ids)
        overlap_id = 0
        if ids is not None:
            for i in range(self.total_positions):
                if i in self.relative_positions and len(self.relative_positions[i]) > 0:
                    diff = np.abs(np.array(corners) - np.array(self.relative_positions[i]))
                    if np.all(diff <= tolerance):
                        overlap_id = i
                        self.update_instruction(f"Overlap detected with position {overlap_id}")
                        cv2.polylines(flipped_frame, np.int32([self.relative_positions[overlap_id]]), True, (0, 255, 0), 4)
                    else:
                        self.update_instruction("Align with blue highlighted square")
                        cv2.polylines(flipped_frame, np.int32([self.relative_positions[overlap_id]]), True, (255, 0, 0), 4)
        flipped_frame = cv2.flip(flipped_frame, 1)
        frame = self.function2(flipped_frame)
        return frame

    def get_dominant_color_hsv(self, image, k=2):
        image = cv2.flip(image, 1)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1)
        kmeans.fit(hsv_pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        dominant_color = colors[label_counts.most_common(1)[0][0]]
        lower_skin = np.array([max(0, dominant_color[0] - 20), 70, 100], dtype=np.uint8)
        upper_skin = np.array([min(179, dominant_color[0] + 20), 255, 255], dtype=np.uint8)
        return lower_skin, upper_skin

    def highlight_navy_blue(self, frame1, lower_skin, upper_skin):
        frame1 = cv2.flip(frame1, 1)
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
        highlighted_frame = frame1.copy()
        highlighted_frame[mask1 != 0] = [0, 0, 255]
        rgb_mask = self.convert_grayscale_to_rgb_mask(mask1)
        return highlighted_frame, rgb_mask

    def convert_grayscale_to_rgb_mask(self, grayscale_mask):
        rgb_mask = cv2.cvtColor(grayscale_mask, cv2.COLOR_GRAY2BGR)
        red_mask = grayscale_mask == 255
        rgb_mask[red_mask] = [0, 255, 0]
        return rgb_mask

    def run(self):
        self.update_video()
        self.root.mainloop()

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MyFUSApp()
    try:
        app.run()
    finally:
        app.cleanup()
