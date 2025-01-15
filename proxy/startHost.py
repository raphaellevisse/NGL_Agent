import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))  # Get current file's directory
sys.path.append(os.path.abspath(os.path.join(parent_dir, '..')))  # Add parent directory

from Agent import Agent
import time
import torch
import ast
    
# Code for the host computer
class HostAgent():
    def __init__(self, headless=False, start_session=True, action_file_path=None, state_file_path=None, image_path=None):
        self.action_file_path = action_file_path
        self.state_file_path = state_file_path
        self.image_path = image_path
        self.agent = Agent(headless=headless, start_session=start_session) # Initialize agent
        
        
        self.start_url = "https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B138657.265625%2C80856.6953125%2C1335.916015625%5D%2C%22crossSectionScale%22:4.45933655284782%2C%22projectionOrientation%22:%5B0.09884308278560638%2C0.9041123986244202%2C-0.4155852496623993%2C0.009988739155232906%5D%2C%22projectionScale%22:12029.259719517953%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22%21720575940612843473%22%2C%22720575940641265549%22%2C%22720575940625693080%22%2C%22720575940645528430%22%2C%22720575940622572010%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D"


        self.start_reading_session() # Begin the NGL sesion with the URL
        self.write_state() # Write the initial state to the host_states.txt file
    
    def start_reading_session(self):
        """Starts the session on the host machine."""
        self.agent.chrome_ngl.start_neuroglancer_session()
        self.agent.chrome_ngl.change_url(self.start_url)
    
    def write_state(self):
        """Prepares the state for the host machine."""
        save_path = self.image_path
        pos_state = self.agent.get_state()
        # ask chrome to save the image in the right space
        self.agent.chrome_ngl.write_screenshot_bytes(save_path)
        with open(self.state_file_path, 'w') as file:
            print("New pos_state read post-action: ",str(pos_state))
            file.write(str(pos_state))


    def readAction(self):
        """From the host, reading the action log that was written by the cluster."""
        with open(self.action_file_path, "r") as f:
            action = f.read()
            print("Action read from log: ", action)
            return action
    
    def respond(self):
        # Obtain the new written action (output vector as a string)
        try_number = 0
        success = False
        while try_number < 100 and success == False:
            try:
                self.nextAction = self.readAction()
                
                # Apply the new action
                print("Applying action: ", self.nextAction)
                self.agent.apply_actions(ast.literal_eval(self.nextAction)) # Convert nextAction, str -> list, apply to agent
                print("applied action")
                # Get the new state information + screenshot, write to file
                self.write_state()
                success = True
            except Exception as e:
                if str(self.nextAction) == "reset":
                    self.agent.chrome_ngl.change_url(self.start_url)
                    self.write_state()
                    success = True
                else:
                    continue
                print("Too fast ! Retrying...")
                try_number += 1
        print("Did number of tries: ", try_number, " and success: ", success)



        

if __name__ == "__main__":
    action_file_path = './proxy/actions.txt'
    state_file_path = './proxy/host_states.txt'
    image_path = './proxy/host_images/screenshot_bytes2'
    host = HostAgent(headless=False, action_file_path=action_file_path, state_file_path=state_file_path, image_path=image_path)

    
    """Checks for modification of action file"""
    last_modified = os.path.getmtime(action_file_path)

    while True:
        current_time_modified = os.path.getmtime(action_file_path)
        if current_time_modified != last_modified:
            print(f"File {action_file_path} has changed.")
            last_modified = current_time_modified    
            host.respond()



        
