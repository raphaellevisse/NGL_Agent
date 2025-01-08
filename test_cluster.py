from selenium import webdriver
from selenium.webdriver.chrome.service import Service

chrome_binary_path = "chrome/chrome-headless-shell-linux64/chrome-headless-shell"
service = Service("chrome/chromedriver-linux64/chromedriver")
options = webdriver.ChromeOptions()
options.add_argument("--headless")

driver = webdriver.Chrome(service=service, options=options)
driver.get("http://example.com")
print(driver.title)
driver.quit()
