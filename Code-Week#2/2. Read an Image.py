# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:12:00 2024

@author: Noopa Jagadeesh
"""

from skimage import io

img = io.imread("COSC/images/Osteosarcoma_01.tif")
print(img.shape)  #y,x,c
#x = Width = 1376
#y = Height = 1104
#Channels = 3 (RGB)

#------------------------scikit-image----------------------------------------
#Some image processing tasks in skimage require floating point image
#i.e pixels to be scaled between values between 0 and 1
from skimage import img_as_float
img2 = img_as_float(img)

#Don't do this to convert image to float
import numpy as np
img3 = img.astype(float)

#Convert back to 8 bit. This is used when we have to convert the images into float
#do some transformation, and then convert it back to 8 bit
from skimage import img_as_ubyte
img_8bit = img_as_ubyte(img2)

#-----------------------OpenCV-----------------------------------------------
import cv2
grey_img = cv2.imread("images/Osteosarcoma_01.tif", 0)
color_img = cv2.imread("images/Osteosarcoma_01.tif", 1)
#images opened using cv2 are numpy arrays
print(type(grey_img)) 
print(type(color_img)) 

#Big difference between skimage imread and opencv is that 
#opencv reads images as BGR instead of RGB.
img_opencv = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)


