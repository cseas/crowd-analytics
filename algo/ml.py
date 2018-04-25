import numpy as np
import os
import requests
from sklearn.linear_model import Ridge
import cv2
import multiprocessing.pool as mpool

def func(image_data):
    response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
    response.raise_for_status()

    analysis = response.json()
    diic = []
    for i in analysis:
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
    return diic


subscription_key = "02726400482345229652709041c698ba"
assert subscription_key

face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"}
params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}


X = np.loadtxt('Xval.txt', dtype=float)
y = np.loadtxt('yval.txt', dtype=int)

RigeModel = Ridge(alpha = 0.001)
RigeModel.fit(X, y)
cap = cv2.VideoCapture(0)
# set the width and height, and UNSUCCESSFULLY set the exposure time
cap.set(3, 1280)
cap.set(4, 1024)
cap.set(15, 0.1)
pool = mpool.ThreadPool(100)
while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        cv2.imshow("input", img)
        # cv2.imshow("thresholded", imgray*thresh2)
        cv2.imwrite("test.bmp", img)  # writes image test.bmp to disk
        image_data = open("test.bmp", "rb").read()
        result = pool.apply_async(func, args=(image_data,))
        try:
            diic = result.get(timeout=10)
        except:
            print("execption occured")

        if (diic != []):
            Yhat = RigeModel.predict(np.array(dic))
            print("Yhat", Yhat)
            rSquare = RigeModel.score(X, y)
            print("R^2", rSquare)

        os.remove("test.bmp")
        key = cv2.waitKey(9)
        if key == 27:  # Esc key
            break
        pool.close()
        pool.join()

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()
# pool.close()
# pool.join()

