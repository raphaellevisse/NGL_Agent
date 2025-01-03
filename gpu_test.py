from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
# Set up Chrome options to run in headless mode
chrome_options = Options()
mail_address = 'pnirlagent@gmail.com'
password = 'secret-password'
width = 1920
height = 1080
chrome_border_height = 87
height += chrome_border_height
chrome_options.add_argument(f"--window-size={width},{height}")
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
#chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration for headless mode

chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("useAutomationExtension", False)
chrome_options.add_experimental_option("excludeSwitches",["enable-automation"])  


chrome_service = Service("chromedriver-mac-arm64/chromedriver")

driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
driver.set_window_position(0, 0)

try:
    driver.get('https://accounts.google.com/Login')
    email_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.ID, "identifierId"))
    )
    email_input.send_keys(mail_address)
    email_input.send_keys(Keys.RETURN)
    password_input = WebDriverWait(driver, 20).until(
        EC.element_to_be_clickable((By.XPATH, "//input[@type='password']"))
    )
    password_input.send_keys(password)
    password_input.send_keys(Keys.RETURN)

    print("Login attempted. Waiting for confirmation...")
    WebDriverWait(driver, 20).until(EC.url_contains("myaccount.google.com"))
    print("Login successful!")
except Exception as e:
    print("An error occurred:", e)

user_url = "http://localhost:8000/client/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B160533.40625%2C80462.75%2C2479.5%5D%2C%22crossSectionScale%22:1.8496565995583267%2C%22projectionOrientation%22:%5B-0.11066838353872299%2C-0.7560726404190063%2C0.10504592210054398%2C0.6364527344703674%5D%2C%22projectionScale%22:31260.083367410043%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22720575940623044103%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D"
driver.get(user_url)
print(driver.current_url)
time.sleep(1)
# Take a screenshot with the adjusted window size
driver.save_screenshot("screenshot2.png")

# Close the browser
driver.quit()
