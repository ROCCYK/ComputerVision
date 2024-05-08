import cv2

def get_centroid(x, y, w, h):
    return (int(x + w/2), int(y + h/2))

def main(video_path):
    # Load the video
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Create a background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history = 1000, varThreshold = 250, detectShadows = False)

    vehicle_tracks = {}  # Dictionary to track vehicles
    vehicle_id = 1  # Unique identifier for each vehicle

    while True:
        # Read the frame
        ret, frame = video.read()
        if not ret:
            break

        # Preprocessing: Apply Gaussian Blur to reduce noise
        frame_blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame_blurred)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_frame_vehicles = {}  # Track vehicles in the current frame

        for contour in contours:
            # Filter out small contours that are not likely to be vehicles
            if cv2.contourArea(contour) > 2000:
                # Get the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate centroid for the current bounding box
                centroid = get_centroid(x, y, w, h)

                match_found = False

                # Check if this vehicle matches with any existing vehicle
                for id, track in vehicle_tracks.items():
                    tracked_centroid = track['centroid']

                    # Calculate distance between centroids
                    distance = cv2.norm(centroid, tracked_centroid)

                    # If distance is small, consider it the same vehicle
                    if distance < 111:
                        match_found = True
                        current_frame_vehicles[id] = {'centroid': centroid}
                        break

                # If no match found, it's a new vehicle
                if not match_found:
                    current_frame_vehicles[vehicle_id] = {'centroid': centroid}
                    vehicle_id += 1  # Increment only when a new vehicle is added

                # Draw a bounding box around the detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Update vehicle tracks with vehicles from the current frame
        vehicle_tracks = current_frame_vehicles

        # Display the cumulative number of vehicles detected
        cv2.putText(frame, 'Cumulative Vehicles Detected: ' + str(vehicle_id - 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the frame with bounding boxes
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'video.mp4'  # Replace with your video file path
    main(video_path)
