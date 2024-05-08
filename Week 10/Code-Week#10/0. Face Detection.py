# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:00:55 2024

@author: Noopa Jagadeesh
"""

import numpy as np
import cv2

#---loading haarcascade detector---
face_detector=cv2.CascadeClassifier('C:/Users/Downloads/COSC/Week#10/models/haarcascade_frontalface_default.xml')
#---Loading the image-----
img = cv2.imread('C:/Users/Downloads/COSC/Week#10/data/face1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_detector.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
  
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()