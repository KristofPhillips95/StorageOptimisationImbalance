import requests

# Fetch the existing IDs from the API
response = requests.get("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/items")
existing_data = response.json()

# Extract the IDs from the existing data and convert to integers
ids_to_delete = [int(item['id']) for item in existing_data]

print(response.text)
# # Delete each ID from the database
# for id in ids_to_delete:
#     data = {"id":id}
#
#     response = requests.delete("https://swdd9r1vei.execute-api.eu-north-1.amazonaws.com/allitems")
#
#     if response.status_code == 200:
#         print(f"Deleted ID: {id}")
#     else:
#         print(f"Failed to delete ID: {id}, Response: {response.text}")
