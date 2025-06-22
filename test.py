import requests

url = "https://phising-detection-server.onrender.com/predict"

headers = {
    "Content-Type": "application/json"
}

# Two sample test cases
test_data = [
    {
        "features": [37, 2, 0, 0, 8, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 12, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 1, 3, 0, 0, 1, 0, 3, 0, 0, 0, 4.010412069, 2.751629167]
    },
    {
        "features": [70, 5, 0, 0, 12, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 26, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 3, 0, 0, 1, 3, 0, 0, 1, 0, 4, 0, 0, 0, 4.089469983, 3.532573258]
    }
]

for i, data in enumerate(test_data, start=1):
    response = requests.post(url, json=data, headers=headers)
    print(f"Test {i} ➜ {response.json()}")
