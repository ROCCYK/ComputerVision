import cv2
import numpy as np

cap = cv2.VideoCapture("highway.mp4")

subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10, detectShadows=True)
#mog2() adapts itself to subtract the background with the last 120 frames. So if
#during the video there is some lighting change or some other changes, the subtractor 
#is going to adapt itself. This function also includes the GaussianFilter and 
#morphological transformation to remove noise and also a function that can detect shadows.This 
#function also works to detect shadows.
#cahnge history=20,Threshold=25
while True:
    _, frame = cap.read()

    mask = subtractor.apply(frame)

    cv2.imshow("Frame", frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
