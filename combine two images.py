import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

# add = img1+img2                                                            # combine the 2 images

# add = cv2.add(img1, img2)                                                  # add the pixel of both images
#cv2.imshow('add', add)

weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)                          # combine 60% of img1 n 40% of img2
cv2.imshow('weighted', weighted)

cv2.waitKey(0)
cv2.destroyAllWindows()
