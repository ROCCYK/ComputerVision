# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:33:41 2024

@author: Noopa Jagadeesh
"""

import skimage
from skimage import io

img1 = io.imread('Downloads/COSC/images/Osteosarcoma_01.tif')


import cv2
img2 = cv2.imread('Downloads/COSC/images/Osteosarcoma_01.tif')

import numpy as np
a=np.ones((5,5))

import pandas as pd
df = pd.read_csv('images/image_measurements.csv')
print(df.head())

from matplotlib import pyplot as plt
plt.imshow(img2)