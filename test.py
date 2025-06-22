import requests

url = "http://127.0.0.1:5000/predict"  # or your deployed Render URL

# Example 1 (legitimate)
features1 = [37,2,0,0,8,0,0,5,0,0,0,0,0,0,0,12,2,0,0,0,0,0,0,2,0,0,3,0,0,1,3,0,0,1,0,3,0,0,0,4.010412069,2.751629167]

# Example 2 (phishing)
features2 = [70,5,0,0,12,0,0,6,0,0,0,0,0,0,0,26,5,0,0,0,0,0,0,2,0,0,3,0,0,1,3,0,0,1,0,4,0,0,0,4.089469983,3.532573258]

for i, f in enumerate([features1, features2], 1):
    response = requests.post(url, json={"features": f})
    print(f"Test {i} ➜", response.json())
