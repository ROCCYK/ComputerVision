# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:26:12 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np

img = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/black_to_white.jpeg", cv2.IMREAD_GRAYSCALE)

_, threshold_binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY) #The first value 
#(captured by _ ) is the threshold used, which in this case is redundant because 
#we're already specifying the threshold value directly (128). The second value is 
#the thresholded image, which is captured in threshold_binary.
_, threshold_binary_inv = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
_, threshold_trunc = cv2.threshold(img, 128, 255, cv2.THRESH_TRUNC) # All the pixel from 0 to 128 will
#be kept the same. All above 128 will be changed to 128
_, threshold_to_zero = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO) #All the pixels below
# 128 will be changed to zero.All above will be kept as it is.
_, threshold_to_zero_inv = cv2.threshold(img, 128, 255, cv2.THRESH_TOZERO_INV) #All the pixels below
# 128 will be kept the same.All above will be changed to zero.

cv2.imshow("Image", img)
cv2.imshow("th binary", threshold_binary)
cv2.imshow("th binary inv", threshold_binary_inv)
cv2.imshow("th trunc", threshold_trunc)
cv2.imshow("th to zero", threshold_to_zero)
cv2.imshow("th to zero inv", threshold_to_zero_inv)
cv2.waitKey(0)
cv2.destroyAllWindows()

