import json
import os
import requests

# free, 20 calls/min, limit 30K calls/month
subscription_key = "bf5951c5f4934e2e90bc11c48ffb57fa"
# premium, 10 calls/sec, no limit, 66 INR / 1000 calls
# subscription_key = "02726400482345229652709041c698ba"
assert subscription_key

face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"}
params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

# load the stored dictionary data from txt file
f = open('analysed.txt', 'r')
dic = json.loads(f.readline())
f.close()

# testing test images with analysed emotion values
indir = '/home/abhijeet/Documents/python/faceapi/test'
for root, dirs, filenames in os.walk(indir):
	for filename in filenames:
		if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
			image_data = open(indir+'/'+filename, "rb").read()
			response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
			response.raise_for_status()
			analysis = response.json()

			for i in analysis:
				flag = 0
				for j,value in i["faceAttributes"]["emotion"].items():
					if(j == "neutral"):
						continue
					# print (j)
					# print (value)
					# print ("max",dic[j]['max'])
					# print ("min",dic[j]['min'])
					if(value > dic[j]['max'] or value < dic[j]['min']):
						print("Not bored")
						flag = 1
						break
				if(flag == 0):
					print("Bored")