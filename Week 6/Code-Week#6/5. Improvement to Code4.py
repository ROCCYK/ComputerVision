# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 21:22:04 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)


while True:
    _, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  
    lower_blue = np.array([38, 86, 0])
    upper_blue = np.array([121, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)# This returns the contours
    #of the boundaries of white object in the mask.
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 5000:
            cv2.drawContours(frame, contour, -1, (0, 255, 0), 3) #-1 will draw all points in the contours


    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()