import time
import numpy as np
import cv2
from random import randint


fgbg = cv2.createBackgroundSubtractorMOG2(history=20,
                                          varThreshold=16,
                                          detectShadows=False)


cap = cv2.VideoCapture(0)
#print("foi") 

while True:
    _, image = cap.read()
    black = np.zeros_like(image)
    fgmask = fgbg.apply(image)
    #contours
    im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        for (i,contour) in enumerate (contours):
            cv2.drawContours(black, contour, 0, (255,255,255), 3)
            (x,y,w,h) = cv2.boundingRect(contour)
            contour_valid = (w >= 40) and (h >= 40)
            if contour_valid:
                cv2.rectangle(image, (x,y), (x+w,y+h), (randint(0,255),randint(0,255),randint(0,255)),2)
            if not contour_valid:
                continue

    cv2.imshow("mask", black)
    cv2.imshow('Camera', image)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
