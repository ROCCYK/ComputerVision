# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 20:33:08 2024

@author: Noopa Jagadeesh
"""
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/Downloads/COSC/Week-5/images/smoothing/puppy.png')
# Define a function for plotting multiple figures
def plot_img(images, titles):
    fig, axs = plt.subplots(nrows = 1, ncols = len(images),
                          figsize = (20, 20))
    for i, p in enumerate(images):
        axs[i].imshow(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
        axs[i].set_title(titles[i])
        axs[i].axis('off')
    plt.show()
# To show a side by side comparison of different filters with different kernel sizes.
for i in range(3,30,8):
    print("with kernel size: "+str(i))
    a_img = cv2.blur(img,(i,i))
    g_img = cv2.GaussianBlur(img,(i,i),0)
    b_img = cv2.bilateralFilter(img,i,75,75)
    images=[img, a_img, g_img, b_img]
    titles=['original image',
          'box filter image',
          'gaussian filter image',
          'Bilateral filter image']
    plot_img(images, titles)