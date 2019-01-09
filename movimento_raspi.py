import time
import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from random import randint


fgbg = cv2.createBackgroundSubtractorMOG2(history=20,
                                          varThreshold=16,
                                          detectShadows=False)



camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
#print("foi") 
# allow the camera to warmup
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    image = rawCapture.array


    fgmask = fgbg.apply(image)
    #contours
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for (i,contour) in enumerate (contours):
            (x,y,w,h) = cv2.boundingRect(contour)
            contour_valid = (w >= 40) and (h >= 40)
            if contour_valid:
                cv2.rectangle(image, (x,y), (x+w,y+h), (randint(0,255),randint(0,255),randint(0,255)),2)
            if not contour_valid:
                continue


    cv2.imshow('Camera', image)
    rawCapture.truncate(0)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
