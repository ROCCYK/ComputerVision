import cv2
import numpy as np

def find_center(x,y,w,h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    x_center = x + x1
    y_center = y + y1
    return x_center, y_center

def main(video_path):
    # Load the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Create a background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history = 1000, varThreshold = 300, detectShadows = False)

    cumulative_vehicle_count = 0  # Cumulative counter for vehicles
    count = []

    while True:
        # Read the frame
        ret, frame = video.read()
        if not ret:
            break

        # Preprocessing: Apply Gaussian Blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame_blurred)

        # morphological operations for noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _, fg_mask = cv2.threshold(fg_mask, 225, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.dilate(fg_mask, np.ones((7,7)), 3)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter out small contours that are not likely to be vehicles
            if cv2.contourArea(contour) > 2000:
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)

                # Draw a bounding box around the detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                center = find_center(x, y, w, h)
                count.append(center)
                cv2.circle(frame, center, 3, (0, 0, 255), -1)

                # Increase cumulative vehicle count when the center passes the line
                for (x, y) in count:
                    if  y<(600 +8) and y>(600-8):
                        cumulative_vehicle_count+=1
                    cv2.line(frame, (25, 600), (1100, 600), (0, 127, 255), 3)
                    count.remove((x, y))

        # Display the cumulative number of vehicles detected
        cv2.putText(frame, 'Cumulative Vehicles Detected: ' + str(cumulative_vehicle_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame with bounding boxes
        cv2.imshow('Frame', frame)
        cv2.imshow("mask", fg_mask)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'video.mp4'  # Replace with your video file path
    main(video_path)
