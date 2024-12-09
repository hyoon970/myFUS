from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np

def draw_bounding_box_and_mask(frame, pixel_set):
    # Create a blank mask with the same dimensions as the frame
    # mask = np.zeros_like(frame)

    # Convert the pixel set (shoulder outline) to a NumPy array
    pixel_points = np.array(pixel_set, dtype=np.int32)

    # Draw the pixel set on the mask
    # cv.polylines(mask, [pixel_points], isClosed=False, color=(255, 255, 255), thickness=1)

    # Find the bounding box for the pixel set
    x, y, w, h = cv.boundingRect(pixel_points)

    # Make a copy of the original frame to preserve the bounding box area
    frame_copy = frame.copy()

    # Optional: Blur the entire frame (you can also make it white instead)
    blurred_frame = cv.GaussianBlur(frame, (21, 21), 0)

    # Apply the bounding box area from the original frame onto the blurred frame
    blurred_frame[y:y+h, x:x+w] = frame_copy[y:y+h, x:x+w]

    # Draw the bounding box (optional if you just want to visualize it)
    cv.rectangle(blurred_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return blurred_frame


def convert_grayscale_to_rgb_mask(grayscale_mask):
    # Create an RGB image by stacking the grayscale mask 3 times (initially, R=G=B)
    rgb_mask = cv.cvtColor(grayscale_mask, cv.COLOR_GRAY2BGR)

    # Create a binary mask where grayscale values are 255 (foreground)
    red_mask = grayscale_mask == 255

    # Set Red channel to 255 for the 255 pixels (leaving G and B as 0)
    rgb_mask[red_mask] = [0, 0, 255]  # OpenCV uses BGR, so [B=0, G=0, R=255] is red

    return rgb_mask


parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='sil-detect-2.JPG')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

i = 0

totalframes = 20

# frame1 = np.zeros((480, 640))
global fgMask1
fgMask1 = np.zeros((480, 640))

while i < totalframes:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    # print(f'Size of fgMask1 {fgMask1.shape}')
    # print(f'Size of fgMask {fgMask.shape}')
    # print(f'Value of fgMask {fgMask[100:110, 80:100]}')
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # grayfgMask = cv.cvtColor(fgMask, cv.COLOR_BGR2GRAY)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # frame1 = frame1 + grayframe
    fgMask[fgMask < 128] = 0
    fgMask1 = fgMask1 + fgMask

    i = i + 1

    # print('I have reached here')

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

fgMask1 = fgMask1/totalframes
fgMask1[fgMask1<20] = 0 # this threshold seems to work for getting rid of background pixels even with complex
# backgrounds
# fgMask1[fgMask1<128] = 0
# print(f'size of frame capture {np.size(frame1)}')
# cv.imshow('Final frame', frame1/50)
# print(f'Size of fgMask1 {fgMask1.shape}')
# print(f'Printing one row of fgMask1 {fgMask1[1,:]}')

cv.imshow('FG Mask Final', fgMask1)

# Create a mask of the saved silhouette
# img2gray = cv.cvtColor(fgMask1, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(fgMask1.astype(np.uint8), 1, 255, cv.THRESH_BINARY)
# print(f'size of mask is {np.shape(mask)} and type of data is {type(mask[1,1])}')
rgb_mask = convert_grayscale_to_rgb_mask(mask)


# info = np.iinfo(fgMask1.dtype) # Get the information of the incoming image type
# data = fgMask1.astype(np.float64) / info.max # normalize the data to 0 - 1
# data = 255 * data # Now scale by 255
# img = data.astype(np.uint8)

cv.waitKey(3000)
cv.destroyAllWindows()

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    # Region of Image (ROI), where we want to insert logo
    # roi = frame[-size - 10:-10, -size - 10:-10]
    roi = frame[:, :]

    # Set an index of where the mask is
    roi[np.where(rgb_mask)] = 0
    roi += rgb_mask.astype(np.uint8)
    # Call the function to draw the bounding box and apply the mask
    # output_frame = draw_bounding_box_and_mask(roi, fgMask1)

    cv.imshow('WebCam', frame)
    if cv.waitKey(1) == ord('q'):
        break

cv.waitKey(0)
cv.destroyAllWindows()

# capture = cv2.VideoCapture(0)
#
# while True:
#     # Read image and convert to grayscale
#     ret, image = capture.read()
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Canny edge detection
#     edges = cv2.Canny(blurred, 50, 100)
#
#     # Find contours
#     contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Filter and draw contours
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     cv2.imshow("Shoulder Outline", image)
#     cv2.waitKey(0)
#
# cv.waitKey(0)
# cv.destroyAllWindows()


# import cv2
# import numpy as np
#
#
# def detect_silhouette(image):
#     # Load the image
#     # image = cv2.imread(image)
#
#     # Resize the image to a smaller size (optional, for faster processing)
#     image = cv2.resize(image, (640, 480))
#
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("grayscale", gray)
#
#     # Apply Gaussian blur to smooth the image (helps in background subtraction)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # Apply binary thresholding (you might need to adjust the threshold value)
#     _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
#
#     # Find the contours (silhouette)
#     contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # Create a blank image to draw the silhouette
#     silhouette = np.zeros_like(gray)
#
#     # Draw the largest contour (assumed to be the person) on the blank image
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         cv2.drawContours(silhouette, [largest_contour], -1, (255), thickness=cv2.FILLED)
#
#     # Display the silhouette
#     cv2.imshow("Silhouette", silhouette)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# # Example usage
# capture = cv2.VideoCapture(0)
#
# if not capture.isOpened():
#     print('Unable to open')
#     exit(0)
#
# i = 0
#
# totalframes = 20
#
# # # frame1 = np.zeros((480, 640))
# # fgMask1 = np.zeros((480, 640))
#
# while i < totalframes:
#     ret, frame = capture.read()
#     if frame is None:
#         break
#
#     detect_silhouette(frame)


# image_path = 'sil-detect-2.JPG'  # Replace with the path to your image
# detect_silhouette(image_path)
