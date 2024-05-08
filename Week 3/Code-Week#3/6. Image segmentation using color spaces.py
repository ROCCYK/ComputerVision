# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:33:25 2024

@author: Noopa Jagadeesh
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, measure

img = io.imread('C:/Users/Downloads/COSC/images/segment/color-ball.jpg')
plt.imshow(img)

## Convert RGB to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(hsv, (100,90,90), (120,255,255)) #Blue Color
plt.imshow(mask) # We can see from the result tiny dots from the reflection. This is where deep learning can help

# To remove the dots, let us do dialation followed by erosion 
from scipy import ndimage as nd
closed_mask = nd.binary_closing(mask, np.ones((7,7)))
plt.imshow(closed_mask)

#Convert the mask into labels, where each object is given a unique label
label_image = measure.label(closed_mask)
plt.imshow(label_image) #This returns an RGB where color-coded labels are painted over the image.
#In the example color-ball.jpg because a blue ball is occluded it appaears are two balls giving a total of three labels

#Plot the Overlay
from skimage.color import label2rgb
image_label_overlay = label2rgb(label_image, image=img)
plt.imshow(image_label_overlay) #All  the balls not segmented are not relevent, so they are in black and white. 
#And the marbles in blue are now in different colors. Each color represents unique objects.

#Compute image properties and return them as pandas-compatible tables
#Available regionprops: area, bbox, centroid, convex_area, coords, equivalent_diameter, perimeter,
#max_intensity, solidity and many

props = measure.regionprops_table(label_image, img,
                                  properties = ['label', 'area', 'equivalent_diameter', 'mean_intensity', 'bbox'])
import pandas as pd
df = pd.DataFrame(props)
print(df)

#Draw BB
frame = cv2.rectangle(img, (252,126), (311,198), (0, 255,0),5)
frame = cv2.rectangle(img, (16,210), (172,363), (0, 255,0),5)
frame = cv2.rectangle(img, (192,222), (238,275), (0, 255,0),5)
plt.imshow(frame)




