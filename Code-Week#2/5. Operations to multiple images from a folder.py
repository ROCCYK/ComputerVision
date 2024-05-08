# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:22:21 2024

@author: Noopa Jagadeesh
"""

#Let us load images and perform some action.
#import the opencv library so we can use it to read and process images
import cv2
import glob

#select the path
path = "COSC/images/test/*.png*"
img_number = 1  #Start an iterator for image number.
#This number can be later added to output image file names.

for file in glob.glob(path):
    #print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    #print(a)  #print numpy arrays for each file

#let us look at each file
#    cv2.imshow('Original Image', a)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#Process each image - change color from BGR to RGB.
    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)  #Change color space from BGR to RGB
    cv2.imwrite("COSC/images/test_images/Color_image"+str(img_number)+".jpg", c)
    img_number +=1 
    cv2.imshow('Color image', c)
    cv2.waitKey(1000)  #Display each image for 1 second
    cv2.destroyAllWindows()