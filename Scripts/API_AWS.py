import requests

import requests

data = {
    "id": "1",
    "price": 2,
    "name": "myitem"
}

response = requests.put('https://d5mvov6803.execute-api.eu-north-1.amazonaws.com/items', json=data)
print(response.text)
response = requests.delete('https://d5mvov6803.execute-api.eu-north-1.amazonaws.com/items')
print(response.text)
r = requests.get('https://api.github.com/events')
r = requests.get('https://d5mvov6803.execute-api.eu-north-1.amazonaws.com/items')
r = requests.get('https://api.github.com/events')
print(r)