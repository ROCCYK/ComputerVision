# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 20:44:21 2024

@author: Noopa Jagadeesh
"""

#Resize images using OpenCV
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("COSC/images/RGBY.png", 1)   #Color is BGR not RGB
plt.imshow(img)
#use cv2.resize. Can specify size or scaling factor.
#Inter_cubic or Inter_linear for zooming.
#Use INTER_AREA for shrinking
#Following example zooms by 4 times.

resized = cv2.resize(img, None, fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
cv2.imshow("resized pic", resized)
cv2.waitKey(0)          
cv2.destroyAllWindows() 