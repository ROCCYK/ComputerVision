import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt 
import os 


def SIFT(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'data//tesla.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints = sift.detect(imgGray,None)
    imgGray = cv.drawKeypoints(imgGray,keypoints,imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure() 
    plt.imshow(imgGray)
    plt.show() 

if __name__ == '__main__': 
    SIFT() 