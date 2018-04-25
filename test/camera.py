# import cv2
# cam = cv2.VideoCapture(0)
# s, im = cam.read() # captures image
# for i in range(10000000):
#     cv2.imshow("Test Picture", im) # displays captured image
# cv2.imwrite("test.bmp",im) # writes image test.bmp to disk

import cv2
import os

#capture from camera at location 0
cap = cv2.VideoCapture(0)
#set the width and height, and UNSUCCESSFULLY set the exposure time
cap.set(3,1280)
cap.set(4,1024)
cap.set(15, 0.1)

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cv2.imshow("input", img)
    #cv2.imshow("thresholded", imgray*thresh2)

    cv2.imwrite("test.bmp", img)  # writes image test.bmp to disk
    print("haaye haaye")
    os.remove("test.bmp")
    key = cv2.waitKey(10)
    if key == 27: # Esc key
        break


cv2.destroyAllWindows()
cv2.VideoCapture(0).release()