import cv2

def motion_detection(video_path):
    
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # initializing vehicle tracker
    vehicle_tracker = {}
    next_vehicle_id = 1  # first vehicle that appears, each new vehicle will have a unique id, essentially works as a counter. 

    # Function to check if two bounding boxes overlap
    def check_overlap(box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
    
        # coordinates of the intersection rectangle
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        # If the intersection rectangle has non-zero area, the boxes overlap
        if x_right > x_left and y_bottom > y_top:
            return True
        return False

    vehicle_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # applying background subtraction
        mask = bg_subtractor.apply(frame, learningRate=-1)

        # morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 225, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # draw bounding boxes and track vehicles
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Adjust the threshold to filter out small objects
                x, y, w, h = cv2.boundingRect(contour)
                vehicle_box = (x, y, x + w, y + h)

                # Check if the bounding box overlaps with any existing vehicle
                overlapping_vehicle_id = None
                for vehicle_id, existing_box in vehicle_tracker.items():
                    if check_overlap(vehicle_box, existing_box):
                        overlapping_vehicle_id = vehicle_id
                        break

                if overlapping_vehicle_id is None:
                    # This is a new vehicle, assign a new ID
                    vehicle_tracker[next_vehicle_id] = vehicle_box
                    next_vehicle_id += 1
                else:
                    # Update the existing vehicle's bounding box
                    vehicle_tracker[overlapping_vehicle_id] = vehicle_box

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Vehicle {overlapping_vehicle_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Vehicle Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Total Vehicles Detected: {next_vehicle_id}')
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = "video.mp4"
    motion_detection(video_path)