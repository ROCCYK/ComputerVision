# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:12:52 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np

img1 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/road.jpg")
img2 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/car.jpg")

#Add two images
sum = cv2.add(img1, img2) #148+255=255
cv2.imshow("sum", sum)
cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-----------------Weighted Sum------------
import cv2
import numpy as np

img1 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/road.jpg")
img2 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/car.jpg")
weighted = cv2.addWeighted(img1, 1, img2, 0.5, 0)#First image is given a weight of 1 and second image is given 0.5
cv2.imshow("weighted", weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()

#-----------------Improve the blending by doing thresholding------------
#Remove the white background to improve the results

import cv2
import numpy as np

#img1 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/road.jpg")
img2 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/car.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2_gray, 252, 255, cv2.THRESH_BINARY) # Here  we are putting mask on the car.

sum = cv2.add(img2, img2, mask=mask)
cv2.imshow("mask",mask)
cv2.imshow("img2", img2)
cv2.imshow("sum", sum)
cv2.waitKey(0)
cv2.destroyAllWindows()








