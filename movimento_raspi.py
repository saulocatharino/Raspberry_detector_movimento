import time
import numpy as np
import cv2
from scipy.spatial import distance
from picamera.array import PiRGBArray
from picamera import PiCamera

#lower = np.array([0, 133, 100], dtype = "uint8")
#upper = np.array([255, 173, 127], dtype = "uint8")

#lower = np.array([0, 48, 80], dtype = "uint8")
#upper = np.array([20, 255, 255], dtype = "uint8")

fgbg = cv2.createBackgroundSubtractorMOG2(history=20,
                                          varThreshold=16,
                                          detectShadows=False)



camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
print("foi") 
# allow the camera to warmup
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = rawCapture.array


    fgmask = fgbg.apply(image)
    #contours
    im2, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnt = contours[0]
        cv2.drawContours(fgmask, contours, -1, (0, 255, 0), 3)
    except:
        pass
    if len(contours) > 0:
        # find largest contour in mask, use to compute minEnCircle
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        x_ = int(x - radius)
        y_ = int(y - radius)
        h_ = int(radius * 2)
        w_ = int(radius * 2)


        sss = distance.euclidean((x_,y_),(x_ +w_,y_ +h_))

        if sss>30:
            cv2.rectangle(image, (x_, y_), (x_ + w_, y_ + h_), (254, 255, 0), 2)

    cv2.imshow('Camera', image)
    rawCapture.truncate(0)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break



