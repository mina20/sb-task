import requests

url = "http://127.0.0.1:8000/predict/"

features = [67755.05, 78223.24, 70823.31]

response = requests.post(url, json={"features": features})

print(response.json())
