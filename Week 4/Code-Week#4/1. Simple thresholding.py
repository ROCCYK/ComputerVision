# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:01:47 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np

img2 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/car.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2_gray, 0, 255, cv2.THRESH_BINARY)# In binary thresholdng
# all pixels value above threshold 0 will  
# be set to 255. All pixels less than and equal to zero will be set to 0.
cv2.imshow("img2", img2)
cv2.imshow("img2gray", img2_gray)
cv2.imshow("threshold", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

