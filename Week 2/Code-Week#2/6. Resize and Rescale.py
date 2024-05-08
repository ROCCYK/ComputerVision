# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:17:28 2024

@author: Noopa Jagadeesh
"""

# Basic image transformation tasks: resize, rescale, downscale.


import matplotlib.pyplot as plt
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import img_as_ubyte

img = io.imread("COSC/images/Osteosarcoma_01.tif", as_gray=True)# when we 
#use scikit-image to read an image and do a operation to convert to grey, then the pixel values
#become float64 with values b/w 0 to 1.
plt.imshow(img)
plt.imshow(img, cmap='gray')

#Rescale, resize image by a given factor. While rescaling image
img_rescaled_without_aa = rescale(img, 1.0 / 4.0, anti_aliasing=False)  #Check rescales image size in variable explorer 
plt.imshow(img_rescaled_without_aa)

img_rescaled_with_aa = rescale(img, 1.0 / 4.0, anti_aliasing=True)
plt.imshow(img_rescaled_with_aa)

#------------------------------------------------------------------------------
#Resize image to given dimensions (shape)
img_resized = resize(img, (200, 200), anti_aliasing=True) #Check dimensions in variable explorer
plt.imshow(img_resized) # Resized image looks compressed

#------------------------------------------------------------------------------
#Downscale using local mean of elements of each block defined by user
img_downscaled = downscale_local_mean(img, (4, 3))
plt.imshow(img_downscaled)