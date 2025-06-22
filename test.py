import requests

data = {
  "features": [37,2,0,0,8,0,0,5,0,0,0,0,0,0,0,12,2,0,0,0,0,0,0,2,0,0,3,0,0,1,3,0,0,1,0,3,0,0,0,4.010412069,2.751629167]
}

res = requests.post("http://127.0.0.1:5000/predict", json=data)
print(res.json())
