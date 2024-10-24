import requests

url = 'http://localhost:9696/predict'

client = {"job": "student", "duration": 280, "poutcome": "failure"}
res = requests.post(url, json=client)

print(res.json())