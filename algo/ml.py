import numpy as np
import os
import requests
from sklearn.linear_model import Ridge
import cv2
import time
from collections import deque
#import multiprocessing.pool as mpool
import threading

# check for Python version to decide which queue module to import
import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    from Queue import Queue
else:
    from queue import Queue

# creating arrays to save in graphx and graphy text files
graphXarr = np.empty((0, 1), int)
graphYarr = np.empty((0, 1), int)

#def test():
que = Queue()
array_rec = []
yhatf=2
w1=-1
t1=-1
l1=-1
h1=-1

cap = cv2.VideoCapture(0)
exit = 0

def waste_facerec(img , array_rec):
    for rec in array_rec:
        w1 = rec["w1"]
        t1 = rec["t1"]
        h1 = rec["h1"]
        l1 = rec["l1"]
        yhatf = rec["yhat"]

        if yhatf > 0:
            cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (0, 0, 255), 2)
    return img

def facerec(img):
    print("w1=", w1, "t1=", t1, "l1=", l1, "h1=", h1)
    if yhatf > 0:
        cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (255, 0, 0), 2)
    else:
        cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (0, 0, 255), 2)
    return img

def writeframe(img):
    out.write(img)

def showframe():
    global cap

    # set the width and height, and UNSUCCESSFULLY set the exposure time
    cap.set(3, 1280)
    cap.set(4, 1024)
    cap.set(15, 0.1)

    while True:
        #i = i + 1
        #print i
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        #print img
        # frame show function
        # cv2.imshow("thresholded", imgray*thresh2)
        cv2.imshow("input", img)
        global que
        # print("size=", que.qsize())
        # writes image test.bmp to disk
        que.put(img)
        #done = time.time()

        key = cv2.waitKey(10)
        if key == 27:  # Esc key
            break
    cv2.destroyAllWindows()
    cv2.VideoCapture(0).release()
    print("finished camera")
    global exit
    exit = 1

def func(image_data):
    print("in func function")

    response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
    #print response
    response.raise_for_status()
    analysis = response.json()
    #print "hi"

    print("func")

    diic = []
    video = []
    for i in analysis:
        print(i)
        video.append({"faceRectangle":i["faceRectangle"]})
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
    #print time.time()
    return diic , video

# api code
subscription_key = "02726400482345229652709041c698ba"
assert subscription_key

face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"}
params = {
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

# scikit code
X = np.loadtxt('Xval.txt', dtype=float)
y = np.loadtxt('yval.txt', dtype=int)

RigeModel = Ridge(alpha = 0.001)
RigeModel.fit(X, y)
#pool = mpool.ThreadPool(10)
thread1 = threading.Thread(target=showframe, args=())
#t2 = threading.Thread(target=test, args=())
thread1.start()
#t2.start()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
framecount=0

i=50
while True:
    #try:
        #print "testing"
        #done = time.time()
        #print "size=", que.qsize()

        if que.qsize() >= 16:
            img = que.get()

            if not que.empty():

                print("----------------------------------------")
                print(que.qsize())

                print("----------------------------------------")
                #img = que.get()
                cv2.imwrite("test.bmp", img)

                # store time of capturing frame in graphx.txt
                graphXarr = np.insert(graphXarr, len(graphXarr), time.time())

                image_data = open("test.bmp", "rb").read()

                diic , video = func(image_data)

                print("got diic")

                if (diic != []):
                    Yhat = RigeModel.predict(np.array(diic))
                    count = 0
                    array_rec = []
                    for one in video:
                        rec=one["faceRectangle"]

                        yhatf = Yhat[count]
                        w1 = rec["width"]
                        t1 = rec["top"]
                        h1 = rec["height"]
                        l1 = rec["left"]
                        array_rec.append({"w1":w1 ,"t1":t1 ,"h1":h1 ,"l1":l1, "yhat":yhatf })
                        img = facerec(img)
                        count = count + 1

                    #framecount+=1
                    writeframe(img)

                    #print "framecount",framecount

                    print("Yhat", Yhat)
                    rSquare = RigeModel.score(X, y)
                    print("R^2", rSquare)

                #cv2.rectangle(img, (50, 50), (50 + 50, 50 + 50), (255, 0, 0), 2)

                #out.write(img)
                if w1 != -1:
                    for cc in range(15):
                        img = que.get()
                        waste_facerec(img,array_rec)
                        writeframe(img)

                os.remove("test.bmp")
                #print video
        else:
            #print "queue is empty"
            if exit == 1:
                print("process finished")
                break

    #except:
        #w = 0
        #print("exception occured")
thread1.join()
out.release()

# save created numpy arrays in respective text files to create graph
np.savetxt('graphx.txt', graphXarr, fmt='%d')
# np.savetxt('graphy.txt', graphYarr, fmt='%d')