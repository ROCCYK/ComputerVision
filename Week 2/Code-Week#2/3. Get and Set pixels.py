# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:11:47 2024

@author: Noopa Jagadeesh
"""

import cv2
image = cv2.imread('COSC/images/monalisa.jpg')

#Q1. Load the image, grab its spatial dimensions (width and height), and then display the original image to our screen
(h, w) = image.shape[:2]
cv2.imshow("Original", image)
cv2.waitKey()
cv2.destroyAllWindows()

#Q2. Accesses the pixel located at (0, 0), which is the top-left corner of the image.
(b, g, r) = image[0, 0]
print("Pixel at (0, 0) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

#Q3. Access the pixel located at x=50, y=20
(b, g, r) = image[20, 50]
print("Pixel at (50, 20) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

#Q4. Update the pixel at (50, 20) and set it to red
image[20, 50] = (0, 0, 255)
(b, g, r) = image[20, 50]
print("Pixel at (50, 20) - Red: {}, Green: {}, Blue: {}".format(r, g, b))

#Q5. Compute the center (x, y)-coordinates of the image
#This is accomplished by simply dividing the width and height by two, ensuring integer conversion (since we cannot access “fractional pixel” locations).
(cX, cY) = (w // 2, h // 2)

#Q6. Grab the top-left corner of the image
tl = image[0:cY, 0:cX]
cv2.imshow("Top-Left Corner", tl)
cv2.waitKey()
cv2.destroyAllWindows()

#Q7. Accessing the top-left corner of the image and setting it to green
image[0:cY, 0:cX] = (0, 255, 0)
# Show our updated image
cv2.imshow("Updated", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
