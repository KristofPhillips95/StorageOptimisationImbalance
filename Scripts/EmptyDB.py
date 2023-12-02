import requests

# Fetch the existing IDs from the API
api_link = "https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items"
response = requests.get(api_link)
existing_data = response.json()

# Extract the IDs from the existing data and convert to integers
ids_to_delete = [int(item['id']) for item in existing_data]

#print(response.text)
print("Deleting items with id:" , ids_to_delete)
# Delete each ID from the database
for id in ids_to_delete:
    response = requests.delete(f"https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items/{id}")
    print(response)
    if response.status_code == 200:
        print(f"Deleted ID: {id}")
    else:
        print(f"Failed to delete ID: {id}, Response: {response.text}")
