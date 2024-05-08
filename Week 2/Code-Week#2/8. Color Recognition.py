# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:23:09 2024

@author: Noopa Jagadeesh
"""

import cv2 
path = r'COSC\images\red.png'  #Try blue.png
image = cv2.imread(path) 
print(image)
# Window name in which image is displayed 
window_name = 'image'
# Using cv2.imshow() method 
# Displaying the image 
cv2.imshow(window_name, image) 
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
# closing all open windows 
cv2.destroyAllWindows()
