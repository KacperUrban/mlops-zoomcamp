import requests

ride = {
    'PULocationID': 10,
    'DOLocationID': 50,
    'trip_distance': 40
}
url = 'http://127.0.0.1:9696/predict'
pred = requests.post(url, json=ride)

print(pred.json())