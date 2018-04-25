import json
import os
import requests

# free, 20 calls/min, limit 30K calls/month
subscription_key = "bf5951c5f4934e2e90bc11c48ffb57fa"
# premium, 10 calls/sec, no limit, 66 INR / 1000 calls
# subscription_key = "02726400482345229652709041c698ba"
assert subscription_key


face_api_url = 'https://southeastasia.api.cognitive.microsoft.com/face/v1.0/detect'

# image_url = 'https://how-old.net/Images/faces2/main007.jpg'
# image_data = open(image_path, "rb").read()

headers = { 'Ocp-Apim-Subscription-Key': subscription_key, "Content-Type": "application/octet-stream"}
params = {
    'returnFaceId': 'false',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'emotion',
}

dic = {
	'sadness':{'min':2.0,'max':0.0},
	'contempt':{'min':2.0,'max':0.0},
	'anger':{'min':2.0,'max':0.0},
	'disgust':{'min':2.0,'max':0.0},
	'surprise':{'min':2.0,'max':0.0},
	'fear':{'min':2.0,'max':0.0},
	'happiness':{'min':2.0,'max':0.0},
	'neutral':{'min':2.0,'max':0.0}
}

# analysing images folder to find ranges of emotions for bored expression
indir = '/home/abhijeet/Documents/python/faceapi/boredImages'
for root, dirs, filenames in os.walk(indir):
	for filename in filenames:
		if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"): 
	    	# print(os.path.join(directory, filename))
			image_data = open(indir+'/'+filename, "rb").read()
			response = requests.post(face_api_url, params=params, headers=headers, data=image_data)
			response.raise_for_status()
			analysis = response.json()
			
			for i in analysis:
				for j,value in i["faceAttributes"]["emotion"].items():
					
					if(value > dic[j]['max']):
						dic[j]['max'] = i["faceAttributes"]["emotion"][j]
					if(value < dic[j]['min']):
						dic[j]['min'] = i["faceAttributes"]["emotion"][j]
					#print(i["faceAttributes"]["emotion"]["contempt"])
			
		else:
			print('invalid image')
	print(dic)

f = open('analysed.txt', 'w')
f.write(json.dumps(dic))
f.close()
#faces = response.json()
#HTML("<font size='5'>Detected <font color='blue'>%d</font> faces in the image</font>"%len(faces))