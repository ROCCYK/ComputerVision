import cv2 as cv 
import numpy as np 
import os 
import matplotlib.pyplot as plt 


def colorHistogram(): 
    root = os.getcwd()
    imgPath = os.path.join(root,'images\\cat.jpg')
    img = cv.imread(imgPath)

    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    
    colors = ['b','g','r']

    plt.figure()
    for i in range(len(colors)): 
        hist = cv.calcHist([imgRGB], [i], None, [256], [0, 256])
        plt.plot(hist,colors[i])
    plt.xlabel("pixel intensity")
    plt.ylabel("# of Pixels")
    plt.show()



if __name__ == '__main__': 
    colorHistogram() 