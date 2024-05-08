# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:40:57 2024

@author: Noopa Jagadeesh
"""

import os
import argparse

import cv2
import mediapipe as mp
import matplotlib as plt
# read image
image_path = 'C:/Users/Downloads/COSC/Week#9/face.png'
img = cv2.imread(image_path)
H, W, _ = img.shape

# detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    #print(out.detections)
    
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            print(x1, y1, w, h)
            
            img =cv2.rectangle(img, (x1,y1), (x1+w, y1+h), (0, 255,0), 10)
            
    cv2.imshow('img',img)
    cv2.waitKey(0)