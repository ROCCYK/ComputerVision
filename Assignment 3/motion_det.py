import cv2
import numpy as np
def motion_detection(video_path):
    
    cap = cv2.VideoCapture(video_path)

    # Initialize background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # initializing vehicle tracker
    vehicle_tracker = {}
    next_vehicle_id = 0  # first vehicle that appears, each new vehicle will have a unique id, essentially works as a counter. 
    
    offset=4
    
    count_line_position = 200

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
    
    def centre_finder(x, y, w, h):
        x1 = int(w/2)
        y1 = int(h/2)
        cx = x+x1
        cy = y+y1
        return cx, cy 
    
    counter = []
    
    vehicle_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # grey and blur the image
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        # applying background subtraction
        mask = bg_subtractor.apply(frame, learningRate=-1)

        # morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _, mask = cv2.threshold(mask, 225, 255, cv2.THRESH_BINARY)
        # mask = cv2.dilate(mask, np.ones((7,7)), 3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of moving objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.line(frame, (175, count_line_position), (800, count_line_position), (255, 127, 0), 3)
        cv2.imshow("frame", mask)
        
        # height, width, _ = frame.shape
        # print(height, width)

        # draw bounding boxes and track vehicles
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2500:  # Adjust the threshold to filter out small objects
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
                center = centre_finder(x, y, w, h)
                counter.append(center)
                cv2.circle(frame, center, 3, (0,0,255), -1)
                
                for (x, y) in counter:
                    if  y<(count_line_position + offset) and y>(count_line_position-offset):
                        vehicle_counter+=1
                    cv2.line(frame, (25, count_line_position), (1100, count_line_position), (0, 127, 255), 3)
                    counter.remove((x,y))
                    # print("Vehicle Counter:" +str(vehicle_counter))
                

        cv2.putText(frame, "Vehicle Counter: "+str(vehicle_counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
        
        # Display the processed frame
        cv2.imshow('Vehicle Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f'Total Vehicles Detected: {vehicle_counter}')
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = "video.mp4"
    motion_detection(video_path)
    
    
