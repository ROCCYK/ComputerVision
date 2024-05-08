# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:11:21 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np

img1 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/road.jpg")
img2 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/car.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2_gray, 252, 255, cv2.THRESH_BINARY) # Here  we are putting mask on the car.
sum = cv2.add(img2, img2, mask=mask)

cv2.imshow("img1", img1)
cv2.imshow("sum", sum)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--------------------------Blend the Images------------------------------------
import cv2
import numpy as np

img1 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/road.jpg")
img2 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/car.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2_gray, 252, 255, cv2.THRESH_BINARY) 
road = cv2.bitwise_and(img1, img1, mask=mask)#bitwise operation takes road image and put the mask on 
#the road image

cv2.imshow("road_background", road)
cv2.waitKey(0)
cv2.destroyAllWindows()

#--------------------------Blend the Images------------------------------------
import cv2
import numpy as np

img1 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/road.jpg")
img2 = cv2.imread("C:/Users/Downloads/COSC/Week-4/images/car.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2_gray, 252, 255, cv2.THRESH_BINARY) #change 252 to 240
mask_inv = cv2.bitwise_not(mask)

road = cv2.bitwise_and(img1, img1, mask=mask)
car = cv2.bitwise_and(img2, img2, mask=mask_inv)
result = cv2.add(road, car)

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("road background", road)
cv2.imshow("car no background", car)
cv2.imshow("mask", mask)
cv2.imshow("mask inverse", mask_inv)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()