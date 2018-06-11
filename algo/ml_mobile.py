import numpy as np
import os
import requests
from sklearn import svm
import cv2
import time
import threading
import sys
import urllib


# check for Python version to decide which queue module to import
is_py2 = sys.version[0] == '2'
if is_py2:
    from Queue import Queue
else:
    from queue import Queue


img11=urllib.request.urlopen('http://192.168.14.187:8080/shot.jpg')
imgnp=np.array(bytearray(img11.read()),dtype=np.uint8)
imgcv=cv2.imdecode(imgnp,-1)
image_height, image_width, channels = imgcv.shape
def getcvimg():
    img11=urllib.request.urlopen('http://192.168.14.187:8080/shot.jpg')
    imgnp=np.array(bytearray(img11.read()),dtype=np.uint8)
    imgcv=cv2.imdecode(imgnp,-1)
    #imgcv1 = cv2.resize(imgcv, (80, 24))

    return imgcv

# creating arrays to save in graphx and graphy text files
graphXarr = np.empty((0, 1), int)
graphYarr = np.empty((0, 1), int)

open('test.txt','w')
with open('test.txt','a') as myfile:
    myfile.write("anger contempt disgust fear happiness neutral sadness surprise roll\n")

# create queue to store video frames
que = Queue()
array_rec = []
yhatf=2
w1=-1
t1=-1
l1=-1
h1=-1

cap = cv2.VideoCapture()
#cap = cv2.VideoCapture(0) # for webcam
exit = 0

def waste_facerec(img , array_rec):
    for rec in array_rec:
        w1 = rec["w1"]
        t1 = rec["t1"]
        h1 = rec["h1"]
        l1 = rec["l1"]
        yhatf = rec["yhat"]

        if yhatf > 0:
            cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (255, 0, 0), 2)
    return img

def facerec(img):
    print("w1=", w1, "t1=", t1, "l1=", l1, "h1=", h1)
    if yhatf > 0:
        # if bored, then red rectangle
        with open('test.txt','a') as myfile:
            myfile.write("bored\n")

        cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (0, 0, 255), 2)
    else:
    	# blue rectangle
        with open('test.txt','a') as myfile:
            myfile.write("not bored\n")

        cv2.rectangle(img, (l1, t1), (l1 + w1, t1 + h1), (255, 0, 0), 2)
    return img

def writeframe(img):
    for i in range(6):
        out.write(img)

# function to increase opencv frame brightness
def adjust_gamma(image, gamma=2.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def showframe():
    global cap

    # set the width and height, and UNSUCCESSFULLY set the exposure time
    cap.set(3, image_width)
    cap.set(4, image_height)
    cap.set(15, 0.1)

    while True:
        img = getcvimg();
        #print(img)
        #img = cv2.flip(img, 1)

        #increase brightness
        gamma = 2.7
        img = adjust_gamma(img, gamma=gamma)


        # frame show function
        # cv2.imshow("thresholded", imgray*thresh2)
        cv2.imshow("input", img)
        global que
        # print("size=", que.qsize())
        # writes image test.bmp to disk
        que.put(img)

        key = cv2.waitKey(10)
        if key == 27:  # Esc key
            print("Escape key pressed")
            break
    cv2.destroyAllWindows()
    cap.release()
    #cv2.VideoCapture().release()
    print("finished camera")
    global exit
    exit = 1

def func(image_data):
    print("in func function")

    response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
    #print response
    response.raise_for_status()
    analysis = response.json()

    print("func")

    diic = []
    video = []
    for i in analysis:
        with open('test.txt','a') as myfile:
            
            print(i)
            video.append({"faceRectangle":i["faceRectangle"]})
            dic = []
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["anger"])
            myfile.write("anger="+str(i["faceAttributes"]["emotion"]["anger"])+"\t")
        
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["contempt"])
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["disgust"])
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["fear"])
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["happiness"])
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["neutral"])
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["sadness"])
            dic.insert(len(dic), i["faceAttributes"]["emotion"]["surprise"])
            #dic.insert(len(dic), i["faceAttributes"]["smile"])
            #dic.insert(len(dic), abs(i["faceAttributes"]["headPose"]["roll"]))

            myfile.write("contempt="+str(i["faceAttributes"]["emotion"]["contempt"])+"\t")
            myfile.write("disgust="+str(i["faceAttributes"]["emotion"]["disgust"])+"\t")
            myfile.write("fear="+str(i["faceAttributes"]["emotion"]["fear"])+"\t")
            myfile.write("happiness="+str(i["faceAttributes"]["emotion"]["happiness"])+"\t")
            myfile.write("neutral="+str(i["faceAttributes"]["emotion"]["neutral"])+"\t")
            myfile.write("sadness="+str(i["faceAttributes"]["emotion"]["sadness"])+"\t")
            myfile.write("surprise="+str(i["faceAttributes"]["emotion"]["surprise"])+"\n")
            #myfile.write("smile="+str(i["faceAttributes"]["smile"])+"\n")
            #myfile.write("roll="+str(abs(i["faceAttributes"]["headPose"]["roll"]))+"\n")
        
            diic.insert(len(diic), dic)
    print(diic)
    

    return diic, video

# api code
subscription_key = "02726400482345229652709041c698ba"
assert subscription_key

face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"}
params = {
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion,smile,headPose'
}

# scikit code
X = np.loadtxt('Xval.txt', dtype=float)
y = np.loadtxt('yval.txt', dtype=int)

clf = svm.SVC()
clf.fit(X, y)
thread1 = threading.Thread(target=showframe, args=())
thread1.start()

frame_width = image_width
frame_height = image_height
print("frame_width=",frame_width,"   ","frame_height=",frame_height)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
framecount=0
let_enter = 0
i=50
while True:
    if que.qsize() >= 20 or let_enter == 1:
        if que.qsize() <=1:
            print("breaking")
            break
        img = que.get()

        if not que.empty():

            print("----------------------------------------")
            print(que.qsize())

            print("----------------------------------------")

            cv2.imwrite("test.bmp", img)


            image_data = open("test.bmp", "rb").read()

            diic, video = func(image_data)

            print("dictionary acquired")

            if (diic != []):
                Xhat = np.array(diic)
                Yhat = clf.predict(Xhat)

                # store time of capturing frame in graphx.txt
                graphXarr = np.insert(graphXarr, len(graphXarr), time.time())

                count = 0
                array_rec = []
                for one in video:
                    rec = one["faceRectangle"]

                    if( (Xhat[count][5] > 0.85) and (Xhat[count][6] > 0.002) ):
                        Yhat[count] = 1

                    # positive value of yhatf means that person can be categorised as bored
                    # correct errors here using Xhat
                    yhatf = Yhat[count]

                    w1 = rec["width"]
                    t1 = rec["top"]
                    h1 = rec["height"]
                    l1 = rec["left"]
                    array_rec.append({"w1":w1 ,"t1":t1 ,"h1":h1 ,"l1":l1, "yhat":yhatf })
                    img = facerec(img)
                    count = count + 1

                # store number of people bored in graphyval temporarily and insert it to numpy array for every frame captured
                graphyval = 0
                for i in Yhat:
                    if i > 0:
                        graphyval += 1
                # insert into numpy array
                graphYarr = np.insert(graphYarr, len(graphYarr), graphyval)

                writeframe(img)

                print("Prediction Array", Yhat)
                mScore = clf.score(X, y)
                print("Model Score", mScore)

            #cv2.rectangle(img, (50, 50), (50 + 50, 50 + 50), (255, 0, 0), 2)

            if w1 != -1 and que.qsize() > 5:
                for cc in range(1):
                    img = que.get()
                    waste_facerec(img,array_rec)
                    #writeframe(img)

            os.remove("test.bmp")

    else:
        #print("queue is empty")
        
        if exit == 1:
            print("exit=1")
            let_enter = 1
            if que.qsize() < 5:
                print("process finished")
                break

thread1.join()
out.release()

# save created numpy arrays in respective text files to create graph
np.savetxt('graphx.txt', graphXarr, fmt='%d')
np.savetxt('graphy.txt', graphYarr, fmt='%d')
