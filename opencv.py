import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
img = cv2.imread('D:\yo.jpg', 1)                                      # to read image , 0 for grey/ 1 for color/-1 for unchang

# cv2.line(img, (0, 0), (150, 150), (255, 255, 255), 15)              # to draw line on image form point to point & cv2 has color in bgr form & width
# cv2.rectangle(img, (15, 25), (200, 150), (0, 255, 0), 5)            # to draw rect on image with start and end points given
# cv2.circle(img, (100, 63), 55, (0, 0, 255), -1)                     # to draw circle with centre,radius given & -1 to fill the circe or any no. for width

# pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)   # take mutiple points in array
# pts = pts.reshape((-1, 1, 2))                                       # to make array as 1*2 array , no need here as it is already 1*2
# cv2.polylines(img, [pts], True, (0, 255, 255), 3)                   # to draw polygon & true whether to join first n last point

# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, 'opencv tuts!', (0, 130), font, 1, (200, 255, 166), 2, cv2.LINE_AA)      # to type text from start point & font & size & color & thickness

img[55, 55] = [255, 255, 255]                                       # chng color of particular point
img[300:350, 100:150] = [255, 255, 255]                             # chng color of region of image(roi)

img_face = img[37:111, 107:194]                                     # copies the roi into img_face
img[0:74, 0:87] = img_face                                          # paste into given roi

cv2.imshow('image', img)                                            # to show image
cv2.waitKey(0)                                                      # wait till we press any key in output screen
cv2.destroyAllWindows()

# plt.imshow(img, cmap='gray', interpolation='bicubic')             # convert image into grey 
# plt.plot([50, 100], [80, 100], 'c', linewidth = 5)                # draw a line
# plt.show()                                                        # show img 

# cv2.imwrite('pj.png', img)                                        # to save our image
'''



'''
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainlogo.png')

rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]                                                   # I want to put logo on top-left corner, So I create a ROI


img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)                            # convert image into grey

ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)         # Now create a mask of logo by threshold , if px>220 then convert to 255 else 0 n take inverse of it

mask_inv = cv2.bitwise_not(mask)                                             # inverse mask by not operator


img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)                           # Now black-out the area of logo in ROI by AND operator


img2_fg = cv2.bitwise_and(img2, img2, mask=mask)                             # Take only region of logo from logo image by AND operator

dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

cv2.imshow('res', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''



'''
cap = cv2.VideoCapture(0)

while (1):

    _, frame = cap.read()

    laplacian = cv2.Laplacian(frame, cv2.CV_64F)                                   # creating gradients of image
    sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

    edges = cv2.Canny(frame, 100, 100)                                             # for edge detection n decrease the dimention for more edges

    cv2.imshow('Original', frame)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)
    cv2.imshow('Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
'''
