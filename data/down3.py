import requests
import json

animal = input('Enter animal: ')

# replace with your own CSE ID and API key
cse_id = "d5917263aacca41f1"
api_key = "AIzaSyCT0YkBhFoZKrfRtk2BxUG3XAAJxI-glIs"

url = f"https://www.googleapis.com/customsearch/v1?q={animal}=1&start=1&searchType=image&key={api_key}&cx={cse_id}"

response = requests.get(url)
response.raise_for_status()

search_results = response.json()
image_url = search_results['items'][1]['link']

print('Image URL:', image_url)