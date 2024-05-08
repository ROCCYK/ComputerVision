# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 01:07:16 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np
original = cv2.imread("C:/Users/Downloads/COSC/Week#10/data/bridge.png")
duplicate = cv2.imread("C:/Users/Downloads/COSC/Week#10/data/bridge-copy.png")

if original.shape == duplicate.shape:
    print("The images have same size and channels")
difference = cv2.subtract(original, duplicate)
b, g, r = cv2.split(difference)
if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print("The images are completely Equal")
    
    
#Here two images are completely equal (same size, same channels, and same pixels values).
#But what if they’re not equal?
#The subtraction method doesn’t work anymore, as we can’t subtract pixels from images that 
#have different sizes, we would get an error.