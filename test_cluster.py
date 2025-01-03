from selenium import webdriver
from selenium.webdriver.chrome.service import Service

service = Service("chromedriver-mac-arm64/chromedriver")
options = webdriver.ChromeOptions()
options.add_argument("--headless")

driver = webdriver.Chrome(service=service, options=options)
driver.get("http://example.com")
print(driver.title)
driver.quit()
