from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from MouseActionHandler import MouseActionHandler
import time
import json
import urllib.parse
from PIL import Image
import io

class ChromeNGL:
    def __init__(self, width: int = 1920, height: int = 1080):
        self.state = None
        self.action = None
        self.reward = None
        self.window_width = 1920
        self.window_height = 1080
        '''Login to Google Account'''
        self.mail_address = 'pnirlagent@gmail.com'
        self.password = 'secret-password'
        
        chrome_options = Options()
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_experimental_option("excludeSwitches",["enable-automation"])  
        chrome_service = Service("chromedriver-mac-arm64/chromedriver")
        self.driver = webdriver.Chrome(service=chrome_service, options=chrome_options)
        self.driver.set_window_position(0, 0)
        self.driver.set_window_size(1920, 1080)
        self.init_url = 'https://accounts.google.com/Login'
        '''---------------------------------'''

        """Action space:"""
        self.action_handler = MouseActionHandler(self.driver)
  
    def start_session(self):
        self.driver.get(self.init_url)
        print(f"Chrome session started. Navigated to URL: {self.init_url}")
        self.google_login()

    def start_neuroglancer_session(self):
        self.change_url("http://localhost:8000/client/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B160533.40625%2C80462.75%2C2479.5%5D%2C%22crossSectionScale%22:1.8496565995583267%2C%22projectionOrientation%22:%5B-0.11066838353872299%2C-0.7560726404190063%2C0.10504592210054398%2C0.6364527344703674%5D%2C%22projectionScale%22:31260.083367410043%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22720575940623044103%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xz-3d%22%7D")
        time.sleep(1)

    def google_login(self):
        try:
            self.driver.get('https://accounts.google.com/Login')
            email_input = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.ID, "identifierId"))
            )
            email_input.send_keys(self.mail_address)
            email_input.send_keys(Keys.RETURN)
            password_input = WebDriverWait(self.driver, 20).until(
                EC.element_to_be_clickable((By.XPATH, "//input[@type='password']"))
            )
            password_input.send_keys(self.password)
            password_input.send_keys(Keys.RETURN)

            print("Login attempted. Waiting for confirmation...")
            WebDriverWait(self.driver, 20).until(EC.url_contains("myaccount.google.com"))
            print("Login successful!")
        except Exception as e:
            print("An error occurred:", e)
    
    def get_url(self):
        return self.driver.current_url
    
    def change_url(self, user_url: str):
        if self.driver:
            self.driver.get(user_url)  
            #print(f"Navigated to URL: {self._url}")
        else:
            print("Driver not initialized. Please start the session first.")
    
    def stop_session(self):
        if self.driver:
            self.driver.quit()
            print("Chrome session stopped.")
        else:
            print("No driver instance found.")
    
    def get_screenshot(self, save_path: str = None):
        screenshot = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))
        #image = image.resize((256, 256))
        # Optionally save to disk
        if save_path:
            image.save(save_path, format='PNG')
        return image
    

    def get_JSON_state(self):
        #browser_url = self.driver.current_url
        """Parse the URL to get the JSON state"""
        script = """
        
        if (window.viewer && window.viewer.state) {
            return JSON.stringify(viewer.state);
        } else {
            return null;
        }
        """
        state = self.driver.execute_script(script)
        return state if state else None

    def change_JSON_state_url(self, json_state):
        """Change the state of the neuroglancer viewer by changing the URL"""
        try:
            #print("Trying to load JSON state...")
            
            json_object = json.loads(json_state)

            serialized_json = json.dumps(json_object)

            encoded_json = urllib.parse.quote(serialized_json)
            new_url = f"http://localhost:8000/client/#!{encoded_json}"
            self.change_url(new_url)
        except Exception as e:
            print("An error occurred:", e)

    def change_JSON_state(self, json_state: str):
        """Change through Neuroglancer's API (restoreState)"""
        try:
            try:
                json_object = json.loads(json_state)
            except json.JSONDecodeError:
                print("Invalid JSON state provided.")
                return
            script = f"""
            viewer.state.restoreState({json.dumps(json_object)});
            """
            self.driver.execute_script(script)
        except Exception as e:
            print("An error occurred:", e)
    

    def mouse_key_action(self, x: float, y: float, action: str, keysPressed:str = "None"):
        self.action_handler.execute_click(x, y, action, keysPressed)


if __name__ == "__main__":
    chrome_ngl = ChromeNGL()
    chrome_ngl.start_session()
    chrome_ngl.start_neuroglancer_session()
    time.sleep(1)
    
    while True:
        try:
            user_input = input("Enter x and y coordinates separated by a comma (or type 'exit' to quit): ").strip()
            if user_input.lower() == 'exit':
                print("Exiting the script...")
                break

            x, y = map(int, user_input.split(','))
            action = input("Enter the action (e.g., 'left_click', 'right_click', 'double_click'): ").strip()
            print(f"Received: x={x}, y={y}, action={action}")
            keysPressed = input("Enter keys to press (separate by commas and with Capital first letter, e.g., 'Shift, Ctrl'): ").strip()
            if not keysPressed:
                keysPressed = "None"  # If no keys are pressed, default to "None"
            
            chrome_ngl.mouse_key_action(x, y, action, keysPressed)
            print(f"Action '{action}' performed at ({x}, {y}).")
            time.sleep(1) 
        except ValueError:
            print("Invalid input. Please enter x and y as integers separated by a comma.")
        except Exception as e:
            print(f"An error occurred: {e}")
    chrome_ngl.stop_session()
