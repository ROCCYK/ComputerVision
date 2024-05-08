import cv2
import numpy as np
import easyocr

# Create a reader to use for OCR
reader = easyocr.Reader(['en'])  # 'en' for English

# Function to detect red color in the image
def detect_red_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    return cv2.countNonZero(mask) > 0  # Return True if red is detected, else False


# Function to detect shapes, focusing on octagons
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 8:  # Check for octagon
            area = cv2.contourArea(cnt)
            if area > 100:  # Minimum area to filter out small shapes
                # Calculate aspect ratio to ensure it's roughly square (for an octagon)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2:  # Aspect ratio range for a rough square
                    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
                    return True
    return False

# Adjusted function to use EasyOCR for text detection
def detect_text(image):
    results = reader.readtext(image)
    for (bbox, text, prob) in results:
        if "STOP" in text.upper():
            return True
    return False

# Main function to process the images
def process_images(image_paths):
    for path in image_paths:
        image = cv2.imread(path)
        
        if image is None:
            print(f"Failed to load {path}.")
            continue
        
        # Check for red color first
        if detect_red_color(image):
            # Proceed with shape detection and OCR on the original image
            if detect_shapes(image) and detect_text(image):  # Both functions work on the original image
                print(f"'STOP' sign detected in {path}")
            else:
                print(f"'STOP' sign not detected in {path}")
        else:
            print(f"'STOP' sign not detected in {path}")


image_paths = ["image1.png", "image2.png", "image3.png", "image4.png"]
process_images(image_paths)
