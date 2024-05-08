import cv2
import numpy as np

img = cv2.imread("C:/Users/Downloads/COSC/Week#10/data/book.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
#surf = cv2.xfeatures2d.SURF_create()
#orb = cv2.ORB_create(nfeatures=1500)

kp = sift.detect(img, None)

#Descriptors are used when we want to compare images
#keypoints_sift, descriptors = sift.detectAndCompute(img, None)
#keypoints_surf, descriptors = surf.detectAndCompute(img, None)
#keypoints_orb, descriptors = orb.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, kp, None)
#img = cv2.drawKeypoints(img, keypoints_sift, None)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

