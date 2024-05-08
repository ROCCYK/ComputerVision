# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:44:39 2024

@author: Noopa Jagadeesh
"""

import matplotlib.pyplot as plt 
import cv2 as cv 
import numpy as np 
import os 


def houghCircleTransform(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'image2.png')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgGray = cv.medianBlur(imgGray,21)

    circles = cv.HoughCircles(imgGray,cv.HOUGH_GRADIENT,dp=1,minDist=50,param1=200,param2=2,minRadius=30,maxRadius=40)
    circles = np.uint16(np.around(circles))

    for circle in circles[0,:]: 
        cv.circle(imgRGB,(circle[0],circle[1]),circle[2],(255,0,255),2)

    plt.figure() 
    plt.imshow(imgRGB)
    plt.show() 
    
if __name__ == '__main__': 
    houghCircleTransform() 