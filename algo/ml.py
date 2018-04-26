import numpy as np
import os
import requests
from sklearn.linear_model import Ridge
import cv2
import time
from collections import deque
#import multiprocessing.pool as mpool
import threading
#def test():
queue = deque([])
cap = cv2.VideoCapture(0)
exit = 0

def showframe():
    global cap

    # set the width and height, and UNSUCCESSFULLY set the exposure time
    cap.set(3, 1280)
    cap.set(4, 1024)
    cap.set(15, 0.1)
    #i = 0
    start= time.time()
    while True:
        #i = i + 1
        #print i
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        #print img
        # frame show function
        # cv2.imshow("thresholded", imgray*thresh2)
        cv2.imshow("input", img)

        # writes image test.bmp to disk



        done = time.time()
        if done - start > 1.0:
            global queue
            queue.append(img)
            start= time.time()
            #print "saved image"
        key = cv2.waitKey(10)
        if key == 27:  # Esc key
            break
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()
    print "finished camera "
    global exit
    exit = 1


def func(image_data):
    print "in func function"
    start = time.time()
    response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
    #print response
    response.raise_for_status()
    analysis = response.json()
    #print "hi"
    done =time.time()
    print "func"
    print done - start
    print "func"
    diic = []
    vedio = []
    for i in analysis:
        print i
        vedio.append({"faceRectangle":i["faceRectangle"]})
        dic = []
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["anger"])
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["contempt"])
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["disgust"])
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["fear"])
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["happiness"])
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["neutral"])
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["sadness"])
        dic.insert(len(dic), i["faceAttributes"]["emotion"]["surprise"])
        diic.insert(len(diic), dic)
    print(diic)
    print time.time()
    return diic , vedio


subscription_key = "02726400482345229652709041c698ba"
assert subscription_key

face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"}
params = {
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}


X = np.loadtxt('Xval.txt', dtype=float)
y = np.loadtxt('yval.txt', dtype=int)

RigeModel = Ridge(alpha = 0.001)
RigeModel.fit(X, y)
#pool = mpool.ThreadPool(10)
t1 = threading.Thread(target=showframe, args=())
#t2 = threading.Thread(target=test, args=())
t1.start()
#t2.start()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while True:
    try:
        #print "testing"
        start = time.time()
        if exit == 1:
            break
        if queue:
            img = queue.popleft()
            cv2.imwrite("test.bmp", img)
            image_data = open("test.bmp", "rb").read()

            diic , vedio = func(image_data)
            done = time.time()
            print done - start
            #print "got diic"
            if (diic != []):
                Yhat = RigeModel.predict(np.array(diic))
                count = 0
                for one in vedio:
                    rec=one["faceRectangle"]
                    yhatf = one["yhat"]
                    x1 = rec["width"]
                    y1 = rec["top"]
                    w1 = rec["height"]
                    h1 = rec["left"]
                    if Yhat[count] >0:
                        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

                out.write(img)

                print("Yhat", Yhat)
                rSquare = RigeModel.score(X, y)
                print("R^2", rSquare)
            os.remove("test.bmp")
            print vedio

    except:
        w = 0
        # print("exception occured")
t1.join()
out.release()
#t2.join()



