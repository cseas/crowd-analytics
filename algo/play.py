# import numpy as np
# import cv2

# cap = cv2.VideoCapture('outpy.avi')
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# for i in range(length):
#     ret, frame = cap.read()
#     winname = "Output"
#     cv2.namedWindow(winname)        # Create a named window
#     cv2.moveWindow(winname, 700,300)  # Move it to (40,30)
#     cv2.imshow(winname, frame)



#     if cv2.waitKey(80) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import os
os.system('totem outpy.avi')