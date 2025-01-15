from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
CHROMEDRIVER_PATH = "/usr/bin/chromedriver"

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


#chrome_service = Service("chromedriver-mac-arm64/chromedriver")

service = Service(CHROMEDRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)

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
    driver.save_screenshot("screenshot1.png")
    print("Login attempted. Waiting for confirmation...")
    #WebDriverWait(driver, 20).until(EC.url_contains("myaccount.google.com"))
    print("Login successful!")
    time.sleep(2)
except Exception as e:
    print("An error occurred:", e)

user_url = "https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B147837.109375%2C60487.99609375%2C34.66416931152344%5D%2C%22crossSectionScale%22:2.0647310999664876%2C%22projectionOrientation%22:%5B-0.12922360002994537%2C-0.8096680045127869%2C0.5709495544433594%2C-0.041899073868989944%5D%2C%22projectionScale%22:15996.052801426149%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22%21720575940625887631%22%2C%22%21720575940629421159%22%2C%22%21720575940619655814%22%2C%22%21720575940619066625%22%2C%22%21720575940624714474%22%2C%22720575940477675264%22%2C%22720575940612770719%22%2C%22720575940599760095%22%2C%22720575940612632543%22%2C%22720575940612769951%22%2C%22720575940612770207%22%2C%22720575940630799277%22%2C%22%21720575940629223484%22%2C%22720575940605894753%22%2C%22720575940591387181%22%2C%22%21720575940631958615%22%2C%22%21720575940634333338%22%2C%22720575940627083080%22%2C%22%21720575940629044854%22%2C%22%21720575940627125842%22%2C%22%21720575940628114045%22%2C%22720575940608102253%22%2C%22%21720575940614004143%22%2C%22%21720575940627637521%22%2C%22%21720575940628400650%22%2C%22%21720575940640316238%22%2C%22%21720575940611855224%22%2C%22%21720575940614572463%22%2C%22%21720575940648980857%22%2C%22%21720575940626487720%22%2C%22720575940626400912%22%2C%22%21720575940611278102%22%2C%22720575940623485843%22%2C%22%21720575940635978638%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%2C%22selection%22:%7B%22position%22:%5B159821.046875%2C73475.15625%2C2885.277587890625%5D%2C%22layers%22:%7B%22Maryland%20%28USA%29-image%22:%7B%22value%22:144%7D%2C%22flywire_v141_m783%22:%7B%22value%22:%22720575940625887631%22%7D%7D%7D%7D"
driver.get(user_url)
print(driver.current_url)
time.sleep(2)


driver.save_screenshot("screenshot2.png")

driver.quit()
