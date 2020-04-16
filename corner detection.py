import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('opencv-corner-detection-sample.jpg')                               # corner detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                         # convert into grey img
gray = np.float32(gray)
corners = cv2.goodFeaturesToTrack(gray, 20, 0.01, 10)                                # detect the img by corners in gray img , no. of points to detect , quality , space bet 2 points
corners = np.int0(corners)                                                           # convet corner point int
for corner in corners:
    x, y = corner.ravel()                                                            # store the dimension of corner point in x , y
    cv2.circle(img, (x, y), 3, 255, -1)                                              # make circle at corner point

cv2.imshow('Corner', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
