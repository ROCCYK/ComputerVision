# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:24:56 2024

@author: Noopa Jagadeesh
"""

"""
Multi-Otsu calculates several thresholds, determined by the number of desired classes.

Liao, P-S., Chen, T-S. and Chung, P-C., 
“A fast algorithm for multilevel thresholding”, 
Journal of Information Science and Engineering 17 (5): 713-727, 2001

"""


import matplotlib.pyplot as plt
import numpy as np

from skimage import data, io, img_as_ubyte
from skimage.filters import threshold_multiotsu

# Read an image
image = io.imread("C:/Users/Downloads/COSC/Week-4/images/BSE_Image.jpg", as_gray=True)
plt.imshow(image, cmap='gray')

# Apply multi-Otsu threshold 
thresholds = threshold_multiotsu(image, classes=5) #This will give us the threshold and we haven't segmented
#the image yet. There are many ways to do this segmentation and one way is to use digitize.

# Digitize (segment) original image into multiple classes based on the threshold we have defined.
#np.digitize assign values 0, 1, 2, 3, ... to pixels in each class.
regions = np.digitize(image, bins=thresholds)
output = img_as_ubyte(regions)  #Convert 64 bit integer values to uint8

plt.imsave("C:/Users/Downloads/COSC/Week-4/images/Otsu_Segmented.jpg", output)


#Let us look at the input image, thresholds on thehistogram and final segmented image
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
# Plotting the original image.
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original')
ax[0].axis('off')
# Plotting the histogram and the two thresholds obtained from
# multi-Otsu.
ax[1].hist(image.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')
# Plotting the Multi Otsu result.
ax[2].imshow(regions, cmap='Accent')
ax[2].set_title('Multi-Otsu result')
ax[2].axis('off')
plt.subplots_adjust()
plt.show()