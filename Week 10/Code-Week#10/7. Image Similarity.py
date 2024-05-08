# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:24:18 2024

@author: Noopa Jagadeesh
"""

'''
How to compare and find similarities between two images when they’re similar but not exactly identical.
For example you can take an Image and compare it with the same image that has different filters applied on.
'''

import cv2
import numpy as np
original = cv2.imread("C:/Users/Downloads/COSC/Week#10/image/bridge.png")
image_to_compare= cv2.imread("C:/Users/Downloads/COSC/Week#10/image/bridge_contrast.png")

#1) Check if two images are equal
if original.shape == image_to_compare.shape:
    print("The images have same size and channels")
    difference = cv2.subtract(original, image_to_compare)
    b, g, r = cv2.split(difference)
    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("The images are completely Equal")
    else:
        print("The images are NOT equal")
        
#2) Check for similarities between the 2 images
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)
print(len(matches))

good_points = []
ratio = 0.6
for m, n in matches:
    if m.distance < ratio*n.distance:
        good_points.append(m)
print(len(good_points))

result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
#result = cv2.drawMatchesKnn(original, kp_1, image_to_compare, kp_2, matches, None)

cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
cv2.imshow("Original", cv2.resize(original, None, fx=0.4, fy=0.4))
cv2.imshow("Image-To-Compare", image_to_compare)
cv2.waitKey(0)
cv2.destroyAllWindows()