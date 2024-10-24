import requests

url = 'http://localhost:9696/predict'

client = {"job": "management", "duration": 400, "poutcome": "success"}
res = requests.post(url, json=client)

print(res.json())