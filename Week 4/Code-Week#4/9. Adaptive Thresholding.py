# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:32:27 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np

img = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/book_page.jpg")

#From this image I need to only take the text and remove the background.
_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY) #change to 155. Then we see only right side of the image
#This is because the right side of the image is brighter and left side is darker.So a single threshold
#value does not work properly with this image. 
#So here we use adaptive thresholding which works with regions of images. So for this image the function
#is going to find the optimal threshold


img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)#Block size should be an odd number
gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 12)

cv2.imshow("Img", img)
cv2.imshow("Binary threshold", threshold)
cv2.imshow("Mean C", mean_c)
cv2.imshow("Gaussian", gaus)
cv2.waitKey(0)
cv2.destroyAllWindows()