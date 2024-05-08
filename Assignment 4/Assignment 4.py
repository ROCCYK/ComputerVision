import cv2
import pytesseract
import numpy as np

# Function to detect red color in the image
def detect_red_color(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper range of red color in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 + mask2

    # Bitwise-AND mask and original image
    red_detection = cv2.bitwise_and(image, image, mask=mask)
    return red_detection

# Function to detect shapes, particularly octagons
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 8:  # Detect octagons
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), -1)

# Function to perform OCR and check for STOP text
def detect_text(image):
    text = pytesseract.image_to_string(image)
    return "STOP" in text

# Main function to process the images
def process_images(image_paths):
    for path in image_paths:
        image = cv2.imread(path)

        # Detect red regions
        red_regions = detect_red_color(image)

        # Detect shapes in red regions
        detect_shapes(red_regions)

        # Perform OCR to confirm 'STOP' text
        if detect_text(red_regions):
            print(f"'STOP' sign detected in {path}")
        else:
            print(f"'STOP' sign not detected in {path}")

# Assuming you have images stored in a list of paths
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
process_images(image_paths)
