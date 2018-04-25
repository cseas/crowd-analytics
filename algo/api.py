import os
import requests
import numpy as np
# from sklearn.linear_model import Ridge

# free, 20 calls/min, limit 30K calls/month
# subscription_key = "bf5951c5f4934e2e90bc11c48ffb57fa"
# premium, 10 calls/sec, no limit, 66 INR / 1000 calls
subscription_key = "02726400482345229652709041c698ba"
assert subscription_key

face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"}
params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

X = np.empty((0, 8), float)
y = np.empty((0, 1), float)

# traverse file
indir = '/home/abhijeet/PycharmProjects/faceapi/algo'
#for reoot, dirs, filenames in os.walk(indir):
for dirs,dirlist,filenames in os.walk("."):
    print(dirs)

    for filename in filenames:
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            # print(os.path.join(directory, filename))
            #print(filename)
            #image_data = open(indir + '/' + filename, "rb").read()
            image_data = open(indir + '/' + dirs.split('/')[1]+'/'+filename, "rb").read()

            response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
            response.raise_for_status()
            analysis = response.json()

            for i in analysis:

                dic = []
                # j = key
                # value = value of emotion
                # for j, value in i["faceAttributes"]["emotion"].items():
                #     dic.insert(len(dic), value)

                dic.insert(len(dic), i["faceAttributes"]["emotion"]["anger"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["contempt"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["disgust"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["fear"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["happiness"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["neutral"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["sadness"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["surprise"])

                arr=np.array(dic)
                # print(dic)
                # print(i["faceAttributes"]["emotion"])
                X = np.vstack([X, arr])
                # print(dirs)
                # print("---",dirs[2:],"----")
                # print(type(dirs))

                if str(dirs[2:]) == "bored ":
                    y = np.insert(y, len(y), 1)
                else:
                    y = np.insert(y, len(y), -1)

    #print(X)
    #print(y)

np.savetxt('Xval.txt', X, fmt='%f')
np.savetxt('yval.txt', y, fmt='%d')
# scikit code
# print(X)
# print(y)
# RigeModel = Ridge(alpha = 0.1)
# RigeModel.fit(X, y)
# Yhat = RigeModel.predict(np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]))
# print("Yhat", Yhat)