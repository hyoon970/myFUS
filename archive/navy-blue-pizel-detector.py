import cv2
import numpy as np
import keyboard
global rgb_mask

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

    lower_navy_blue = np.array([150, 50, 85])  # Actually for red, adjust the lower bound based on your needs
    upper_navy_blue = np.array([255, 245, 255])  # Actually for red, adjust the upper bound based on your needs

    # lower_navy_blue = np.array([100, 75, 85])   # Actually for bluegreen, adjust the lower bound based on your needs
    # upper_navy_blue = np.array([167, 245, 250])  # Actually for bluegreen, adjust the upper bound based on your needs

    # Create a mask for navy-blue pixels
    mask1 = cv2.inRange(hsv, lower_navy_blue, upper_navy_blue)

    # Create a new frame where navy-blue pixels will be highlighted in red
    highlighted_frame = frame1.copy()

    # Replace navy-blue pixels with bright red in the original frame
    highlighted_frame[mask1 != 0] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    rgb_mask = convert_grayscale_to_rgb_mask(mask1)

    return highlighted_frame, rgb_mask


def main():
    # Example usage with OpenCV VideoCapture
    cap = cv2.VideoCapture(0)  # Capture from webcam

    # while True:
    #     # key = cv2.waitKey(0) & 0xFF  # This waits for key press
    #     # if key == ord('m'):
    #     print("Press the m key darling")
    #     if keyboard.read_event().name == 'm':
    #         # Call the function to highlight navy-blue pixels
    #         highlighted_frame, rgb_mask = highlight_navy_blue(cap)
    #         # Release the capture object here
    #         # cap.release()
    #         # cv2.destroyAllWindows()m
    #         break
    #     else:
    #         print("Press the m key")
    #
    # # print("I'm ready to carry onm")
    #
    # cap = cv2.VideoCapture(0)  # Capture from webcam

    highlighted_frame, rgb_mask = highlight_navy_blue(cap)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        roi = frame[:, :]

        # Set an index of where the mask is
        roi[np.where(rgb_mask)] = 0
        roi += rgb_mask.astype(np.uint8)
        # Call the function to draw the bounding box and apply the mask
        # output_frame = draw_bounding_box_and_mask(roi, fgMask1)

        cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()