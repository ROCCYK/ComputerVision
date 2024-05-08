import cv2 as cv 
import numpy as np 
import os 
import matplotlib.pyplot as plt 


def grayHistogram(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'images\\cat.jpg')
    img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE)

    plt.figure()
    plt.imshow(img,cmap='gray')
    
    '''
        cv.calcHist(image,channels,mask,histSize,histRange - computes the histogram
            @param image List of ndarray 
            @param channels List of channel 
            @param mask Numpy array 
            @param histSize List of number of bins used 
            @param histRange List range of values 
    '''
    # 0 inclusive, 256 exclusive
    
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    plt.figure()
    plt.plot(hist)
    plt.xlabel("pixel intensity")
    plt.ylabel("# of pixels")
    plt.show()


if __name__ == '__main__': 
    grayHistogram() 
