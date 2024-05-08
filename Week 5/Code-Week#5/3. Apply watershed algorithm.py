# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 21:22:20 2024

@author: Noopa Jagadeesh
"""

import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np 
import os 

def watershed(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'images/tree1.png')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB) #BGR2RGB, because opencv
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
   
    
    #Step1: Thresholding. Here we wanted to do an initial extraction of the tree
    _,imgThreshold = cv.threshold(img,120,255,cv.THRESH_BINARY_INV)

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.subplot(232)
    plt.imshow(imgThreshold,cmap='gray')
    
    #Step2: Dialation
    kernel = np.ones((3,3),np.uint8)
    imgDilate = cv.morphologyEx(imgThreshold,cv.MORPH_DILATE,kernel)
    plt.subplot(233)
    plt.imshow(imgDilate)
    
    #Step3:Distance Transformation- computes distance from current pixel to nearest 0(black) pixel
    distTrans = cv.distanceTransform(imgDilate,cv.DIST_L2,5)
    plt.subplot(234)
    plt.imshow(distTrans)#Heatmap of how far pixel is from the background
    
    _,distThresh = cv.threshold(distTrans,15,255,cv.THRESH_BINARY)
    plt.subplot(235)
    plt.imshow(distThresh)
    
    #Step4:Connected Components-Finds different regions
    distThresh = np.uint8(distThresh)
    _,labels = cv.connectedComponents(distThresh)
    plt.subplot(236)
    plt.imshow(labels)
    
    #Step5:Watershed Algorithm
    plt.figure() 
    plt.subplot(121)
    labels = np.int32(labels)
    labels = cv.watershed(imgRGB,labels)
    plt.imshow(labels)
    
    plt.subplot(122)
    imgRGB[labels==-1] = [255,0,0] #label=-1 are where the edges will be. We will change its pixel to 255
    plt.imshow(imgRGB)


    plt.show() 



if __name__ == '__main__': 
    watershed() 