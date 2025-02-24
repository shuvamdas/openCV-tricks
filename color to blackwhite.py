import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)                                                    # takes the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')                                     # for video saving
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))                # for video saving
while True:
    ret, frame = cap.read()                                                  # ret take tru/false and frame takes video
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                           # convert video into grey
    out.write(frame)                                                         # to save video
    cv2.imshow('frame', frame)                                               # show original video
    cv2.imshow('gray', gray)                                                 # show grey video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
