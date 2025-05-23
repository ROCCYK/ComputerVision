# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:19:07 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np
import os 

root = os.getcwd()
img = cv2.imread('C:/Users/Downloads/COSC/Week-5/images/signature.png', 0)
r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Original", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((5,5),np.uint8)

dilated = cv2.dilate(img,kernel,iterations = 1)
cv2.imshow("Dilation", dilated)
cv2.imwrite("dilation.png", dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()

eroded = cv2.erode(img,kernel,iterations = 1)
cv2.imshow("Erosion", eroded)
cv2.imwrite("erosion.png", eroded)
cv2.waitKey(0)
cv2.destroyAllWindows()

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow("Opening", opened)
cv2.imwrite("opening.png", opened)
cv2.waitKey(0)
cv2.destroyAllWindows()

closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closing", closed)
cv2.imwrite("closing.png", closed)
cv2.waitKey(0)
cv2.destroyAllWindows()