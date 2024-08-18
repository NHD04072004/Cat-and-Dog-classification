import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib
import time

# Configure WebDriver
driver_path = r"C:\Users\Administrator\PycharmProjects\Cat-and-Dog-classification\data\chromedriver-win64\chromedriver.exe"  # Replace with the actual path to the downloaded driver
# car_list = [
#     # "toyota+RAV4",
#     # "toyota+86",
#     # "toyota+Sienna",
#     "toyota+Camry",
#     # "toyota+C-HR",
#     # "toyota+Corolla+sedan",
#     # "toyota+4Runner",
#     # "toyota+Venza"
# ]
#
# views = {
#     'stock+photos': 90,
#     "front+view": 90,
#     "side+profile": 90,
#     "back+angle+view": 90,
#     "on+the+road": 90,
#     "tailight+view+photoshoot": 20,
#     "headlight+view+photoshoot": 20,
#     "modifications+photoshoot": 90
# }




numOfPics = 1000
# Launch Browser and Open the URL
counter = 0
# for car_model in car_list:
#     for angle, numOfPics in views.items():
# options.add_argument('--user-data-dir=/Users/a970/Library/Application Support/Google/Chrome/Default')
# options.add_argument('--headless')
driver = uc.Chrome()
# driver.minimize_window()

# Create url variable containing the webpage for a Google image search.
# url = "https://www.google.com/search?q={s}&tbm=isch&tbs=sur%3Afc&hl=en&ved=0CAIQpwVqFwoTCKCa1c6s4-oCFQAAAAAdAAAAABAC&biw=1251&bih=568"
url = str(
    "https://www.google.com/search?q=dog&udm=2")
# Launch the browser and open the given url in your webdriver.
# search_query = "Toyota Supra"

# [general], front view, rear view, side profile, back angle view, on the road, tail-lights, headlights
driver.get(url)
time.sleep(5)

# The execute script function will scroll down the body of the web page and load the images.
driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
time.sleep(5)
if (numOfPics > 50):
    for i in range(0, 3):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        time.sleep(5)
elif (numOfPics > 20):
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    time.sleep(5)

# Review the Web Page’s HTML Structure


# We need to understand the structure and contents of the HTML tags and find an attribute that is unique only to images.
img_results = driver.find_elements(By.XPATH, "//img[@class='YQ4gaf']")

image_urls = []
for img in img_results:
    image_urls.append(img.get_attribute('src'))

folder_path = r"C:\Users\Administrator\PycharmProjects\Cat-and-Dog-classification\data\images"  # change your destination path here

modifiedName = "Dog_"

for i in range(numOfPics):
    counter += 1
    urllib.request.urlretrieve(str(image_urls[i]), folder_path + "{0} {1}.jpg".format(modifiedName, counter))

driver.quit()
counter = 0
