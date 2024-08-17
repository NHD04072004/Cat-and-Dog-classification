import bs4
import os
import requests
from bs4 import BeautifulSoup

downloadPath = './Cat/'
os.makedirs(downloadPath, exist_ok=True)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
page = requests.get('https://www.google.com/search?q=cat&udm=2', headers=headers)

# Parse the page content
soup = BeautifulSoup(page.content, 'html.parser')

# Find all image tags
images = soup.find_all('img')

for idx, image in enumerate(images):
    imgData = image.get('src')
    if imgData:
        print(f"Image {idx}: {imgData}")

        if imgData.startswith('http'):
            try:
                response = requests.get(imgData)
                response.raise_for_status()

                filename = os.path.join(downloadPath, f"image_{idx}.jpg")

                with open(filename, 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Failed to download {imgData}: {e}")

