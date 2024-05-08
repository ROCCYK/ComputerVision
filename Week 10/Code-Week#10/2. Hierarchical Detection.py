# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:26:39 2024

@author: Noopa Jagadeesh
"""

import numpy as np
import cv2

face_detector=cv2.CascadeClassifier('C:/Users/Downloads/COSC/Week#10/models/haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('C:/Users/Downloads/COSC/Week#10/models/haarcascade_eye_tree_eyeglasses.xml')

img = cv2.imread('C:/Users/Downloads/COSC/Week#10/data/two-people.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.2, 4)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
#-----roi_gray is the cropped detected face in grayscale
# --- roi_color is the cropped detected face in color
eyes = eye_detector.detectMultiScale(roi_gray)
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()