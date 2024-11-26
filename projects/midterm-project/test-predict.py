import requests

url = "http://0.0.0.0:9696/predict"

climber = {
  "sex": "M",
  "age": 62,
  "leader": False,
  "deputy": False,
  "msolo": False,
  "member_experience": 0,
  "season": 1,
  "host": 1,
  "camps": 4,
  "rope": 0,
  "totmembers": 5,
  "smtmembers": 2,
  "tothired": 7,
  "smthired": 4,
  "nohired": False,
  "stdrte": True,
  "comrte": False,
  "agency_experience": 85,
  "heightm": 8516,
  "himal": 12,
  "region": 2
}

res = requests.post(
  url=url,
  json=climber
)

print(res.json())