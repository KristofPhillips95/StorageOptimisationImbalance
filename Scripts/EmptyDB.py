import requests
from urllib.parse import quote

api_link = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/lts_items"
response = requests.get(api_link)
existing_data = response.json()

ids_to_delete = [str(item['id']) for item in existing_data]

print("Deleting items with id:", ids_to_delete)

for id in ids_to_delete:
    encoded_id = quote(id)
    response = requests.delete(f"{api_link}/{encoded_id}")
    if response.status_code == 200:
        print(f"Deleted ID: {id}")
    else:
        print(f"Failed to delete ID: {id}, Response: {response.text}")

