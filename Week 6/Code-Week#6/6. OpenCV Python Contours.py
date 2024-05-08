# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 23:04:21 2024

@author: Noopa Jagadeesh
"""

import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
import os 


def contours(): 

    # Get and Draw Contours 
    root = os.getcwd() 
    imgPath = os.path.join(root,'contour_detection//tesla.jpg')
    img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
    plt.figure()
    plt.subplot(231)
    plt.imshow(img,cmap='gray')
    

    height,width = img.shape
    scale = 4
    heightScale = int(scale*height)
    widthScale = int(scale*width)
    img = cv.resize(img,(widthScale,heightScale))

    _,thresh = cv.threshold(img,65,255,cv.THRESH_BINARY)
    kernel = np.ones((7,7),np.uint8)
    thresh = cv.dilate(thresh,kernel)

    '''
        cv.RETR_TREE - retrieves all contours and gets the hierarchy of nested contours 
        cv.RETR_LIST - no hierachy 
        cv.RETR_EXTERNAL - only outer contour 
        cv.RETR_CCOMP - only 2 hierachy 

        CHAIN_APPROX_SIMPLE - decides to only store essential points,
        removes any redudant points 
        CHAIN_APPROX_NONE - will store all points 
    '''
    contours,_ = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contours = [contours[0]] #put contours inside a list
    cv.drawContours(img, contours, -1, (135, 157, 54), 5)

    plt.subplot(232)
    plt.imshow(thresh,cmap='gray')
    plt.subplot(233)
    plt.imshow(img,cmap='gray')

    ##--------------------- Contour Features-----------------------------------    
    # center of mass 
    M = cv.moments(contours[0])
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])
    
    plt.subplot(234)
    plt.imshow(img,cmap='gray')
    plt.plot(Cx,Cy,'r*')

    area = cv.contourArea(contours[0])# Find area of contour
    perimeter = cv.arcLength(contours[0],True)#Find the perimeter

    # contour approx 
    # epsilon - max distance from contour to approx contour 
    epsilon = .01*perimeter
    approx = cv.approxPolyDP(contours[0],epsilon,True)
    approx = np.array(approx)
    approx = np.concatenate((approx,approx[:1]),axis=0)#To append the last value
    plt.plot(approx[:,0,0],approx[:,0,1])#(x,y)

    hull = cv.convexHull(contours[0])
    hull = hull[:,0,:]
    hull = np.concatenate((hull,hull[:1]),axis=0)
    plt.subplot(235)
    plt.imshow(img,cmap='gray')
    plt.plot(hull[:,0],hull[:,1],'r-')

    x,y,w,h = cv.boundingRect(contours[0])
    plt.subplot(236)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    plt.imshow(img,cmap='gray')
    plt.show()  
    

if __name__ == '__main__': 
    contours()