import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('opencv-feature-matching-template.jpg', 0)                         #feature matching
img2 = cv2.imread('opencv-feature-matching-image.jpg', 0)
orb = cv2.ORB_create()                                                               # detector we're going to use for the features
kp1, des1 = orb.detectAndCompute(img1, None)                                         # key points and their descriptors with the orb detector.
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)                                # create a BFMatcher object
matches = bf.match(des1, des2)                                                       # we create matches of the descriptors
matches = sorted(matches, key=lambda x: x.distance)                                  # sort matches based on their distances
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)            # draw line bet first 10 matches
plt.imshow(img3)
plt.show()
