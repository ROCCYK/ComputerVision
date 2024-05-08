import cv2
import numpy as np

# Access the Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the webcam's BGR colour space into HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Get the mask of all orange colours from the webcam
    mask = cv2.inRange(hsvImage, (10, 100, 20), (25, 255, 255))

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw rectangle around the contour
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Place Text of Orange when it identifies orange
        cv2.putText(frame, "Orange", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)

    # Show the live feed
    cv2.imshow('frame', frame)
    
    # Press "q" when you want to stop the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
