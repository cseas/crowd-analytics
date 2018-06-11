import os
import requests
import numpy as np

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
    'returnFaceAttributes': 'emotion,smile,headPose'
}

X = np.empty((0, 8), float)
y = np.empty((0, 1), float)


test_folder_name = "bored"
test_file = "bored.txt"

# remove previous test file
open(test_file,'w')
# writes azure analysed values of files in txt file
def analyse(fname,ara):

    with open(test_file,'a') as myfile:
        #myfile.write("anger\tcontempt\tdisgust\tfear\thappiness\tneutral\tsadness\tsurprise\troll\n")
        myfile.write("anger=" + str(ara[0]) + " ")
        myfile.write("contempt=" + str(ara[1]) + " ")
        myfile.write("disgust=" + str(ara[2]) + " ")
        myfile.write("fear=" + str(ara[3]) + " ")
        myfile.write("happiness=" + str(ara[4])+ " ")
        myfile.write("neutral=" + str(ara[5]) + " ")
        myfile.write("sadness=" + str(ara[6]) + " ")
        myfile.write("surprise=" + str(ara[7]) + " ")
        #myfile.write("roll=" + str(ara[8]))
        myfile.write("\n")
        myfile.write(str(fname)+"\n\n")

        # this is criteria for error
        if(False):
        	shutil.move("/home/abhijeet/Documents/github/crowd-analytics/newTrain/" + test_folder_name + "/" + str(fname), 
        	"/home/abhijeet/Documents/github/crowd-analytics/newTrain/errors/" + str(fname))


# dataset folder
indir = '/home/abhijeet/Documents/github/crowd-analytics/algo/'

progress = 0
for dirs,dirlist,filenames in os.walk("."):
    print(dirs)

    for filename in filenames:
        print(filename)
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".PNG") or filename.endswith(".JPEG"):
            print("taken " + filename)
            # print(os.path.join(directory, filename))
            # image_data = open(indir + '/' + filename, "rb").read()
            image_data = open(indir + '/' + dirs.split('/')[1]+'/'+filename, "rb").read()

            response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
            response.raise_for_status()
            analysis = response.json()

            if analysis:
                print("Face detected and done ", progress, "%, analysing", dirs, "currently")

            for i in analysis:

                #print(i)
                dic = []
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["anger"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["contempt"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["disgust"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["fear"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["happiness"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["neutral"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["sadness"])
                dic.insert(len(dic), i["faceAttributes"]["emotion"]["surprise"])
                #dic.insert(len(dic), i["faceAttributes"]["smile"])
                #dic.insert(len(dic), abs(i["faceAttributes"]["headPose"]["roll"]))

                # convert list to numpy array
                arr = np.array(dic)

                # insert next row to main numpy array
                X = np.vstack([X, arr])


                # analyse particular folder
                if str(dirs[2:]) == test_folder_name:
                    analyse(filename, arr)
                # print(dirs)
                # print("---",dirs[2:],"----")
                # print(type(dirs))

                if str(dirs[2:]) == "bored" or str(dirs[2:]) == "openmouth" or str(dirs[2:]) == "sad":
                    y = np.insert(y, len(y), 1)
                else:
                    y = np.insert(y, len(y), -1)

    # increment value = 100 / no. of directories
    progress += 6.66

    #print(X)
    #print(y)

np.savetxt('Xval.txt', X, fmt='%f')
np.savetxt('yval.txt', y, fmt='%d')