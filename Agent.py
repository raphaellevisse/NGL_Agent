from ChromeNGL import ChromeNGL
from utils import parse_action
import time
import json
import torch
from Values import Values

class Agent:
    def __init__(self, model=None, headless=False, start_session: bool = False):
        self.values = Values()
        self.action_history = []
        self.state = None
        self.action = None
        self.reward = None

        self.sleep_time = 1 # time between actions
        if start_session:
            self.chrome_ngl = ChromeNGL(headless=headless)
            self.chrome_ngl.start_session()
        else:
            self.chrome_ngl = None
        
        #self.model = model
    
    def prepare_state(self, image_path=None, verbose=False):
        state = self.chrome_ngl.get_JSON_state()
        json_state = json.loads(state)
        # for now the state we give in just the parsed position, crossSectionScale, projectionOrientation, projectionScale
        position = json_state["position"]
        crossSectionScale = json_state["crossSectionScale"]
        projectionOrientation = json_state["projectionOrientation"]
        projectionScale = json_state["projectionScale"]
        pos_state = [position, crossSectionScale, projectionOrientation, projectionScale]
        curr_image = self.chrome_ngl.get_screenshot(image_path)
        if verbose:
            print("Current state:", pos_state)
        #normalized_pos_state = self.normalize_pos_state(pos_state)
        return pos_state, curr_image, json_state
  

    def decision(self):
        # TO NOT BE USED IN THE FINAL VERSION
        # make a decision based on the current state
        pos_state, curr_image, json_state = self.prepare_state()
        # preprocess the state by making it a vector
        # input = [position, crossSectionScale, projectionOrientation, projectionScale, image]
        # MODEL INPUT = [state, image] (or memory, TBD)

        ## Does not work as everything should be boolean for RL, make boolean increments for each action ? (Only in Deep Q Learning)
        # OUTPUT OF THE MODEL:
        # Actions can be:
        # - left click bool
        # - right click bool
        # - double click bool
        # - x float mouse position
        # - y float mouse position

        # - key Shift bool
        # - key Ctrl bool
        # - key Alt bool

        # JSON actions:
        # JSON_change: bool
        # delta_position: 1x3 float array
        # delta_crossSectionScale: float
        # delta_projectionOrientation: 1x4 float array
        # delta_projectionScale: float

        # output_vector = [
        #                 left_click, right_click, double_click,  # 3 booleans
        #                 x, y,                                  # 2 floats for mouse position
        #                 key_Shift, key_Ctrl, key_Alt,          # 3 booleans for keys
        #                 json_change,                           # 1 boolean for JSON change
        #                 delta_position_x, delta_position_y, delta_position_z,  # 3 floats
        #                 delta_crossSectionScale,               # 1 float
        #                 delta_projectionOrientation_q1, delta_projectionOrientation_q2,
        #                 delta_projectionOrientation_q3, delta_projectionOrientation_q4,  # 4 floats
        #                 delta_projectionScale                  # 1 float
        #             ]



        discrete_probs, continuous_probs = self.model.action(pos_state, curr_image)
        output_vector = self.model.build_output_vector(discrete_probs, continuous_probs)
        # APPLY ACTIONS
        self.apply_actions(output_vector, json_state)
        return discrete_probs, continuous_probs, output_vector
        
    def apply_actions(self, output_vector, json_state):
      
        (
            left_click, right_click, double_click,  # 3 booleans
            x, y,                                  # 2 floats for mouse position
            key_Shift, key_Ctrl, key_Alt,          # 3 booleans for keys
            json_change,                           # 1 boolean for JSON change
            delta_position_x, delta_position_y, delta_position_z,  # 3 floats
            delta_crossSectionScale,               # 1 float
            delta_projectionOrientation_q1, delta_projectionOrientation_q2,
            delta_projectionOrientation_q3, delta_projectionOrientation_q4,  # 4 floats
            delta_projectionScale                  # 1 float
        ) = [v.item() if isinstance(v, torch.Tensor) else v for v in output_vector]
        # fitting output_vector back into action space
        x = x * self.x_factor
        y = y * self.y_factor
        key_pressed = ""
        if key_Shift:
            print("Shift key pressed")
            key_pressed += "Shift, "
        if key_Ctrl:
            print("Ctrl key pressed")
            key_pressed += "Ctrl, "
        if key_Alt:
            print("Alt key pressed")
            key_pressed += "Alt, "
        key_pressed = key_pressed.strip(", ")

        if left_click:
            print("Decided to do a left click at position", x, y)
            self.chrome_ngl.mouse_key_action(x, y, "left_click", key_pressed)
        elif right_click:
            print("Decided to do a right click at position", x, y)
            self.chrome_ngl.mouse_key_action(x, y, "right_click", key_pressed)
        elif double_click:
            print("Decided to do a double click at position", x, y)
            self.chrome_ngl.mouse_key_action(x, y, "double_click", key_pressed)
        elif json_change:
            print("Decided to change the JSON state")
            json_state["position"][0] += delta_position_x.item() if isinstance(delta_position_x, torch.Tensor) else delta_position_x
            json_state["position"][1] += delta_position_y.item() if isinstance(delta_position_y, torch.Tensor) else delta_position_y
            json_state["position"][2] += delta_position_z.item() if isinstance(delta_position_z, torch.Tensor) else delta_position_z
            
            json_state["crossSectionScale"] += delta_crossSectionScale.item() if isinstance(delta_crossSectionScale, torch.Tensor) else delta_crossSectionScale
            
            json_state["projectionOrientation"][0] += delta_projectionOrientation_q1.item() if isinstance(delta_projectionOrientation_q1, torch.Tensor) else delta_projectionOrientation_q1
            json_state["projectionOrientation"][1] += delta_projectionOrientation_q2.item() if isinstance(delta_projectionOrientation_q2, torch.Tensor) else delta_projectionOrientation_q2
            json_state["projectionOrientation"][2] += delta_projectionOrientation_q3.item() if isinstance(delta_projectionOrientation_q3, torch.Tensor) else delta_projectionOrientation_q3
            json_state["projectionOrientation"][3] += delta_projectionOrientation_q4.item() if isinstance(delta_projectionOrientation_q4, torch.Tensor) else delta_projectionOrientation_q4
            
            json_state["projectionScale"] += delta_projectionScale.item()*50 if isinstance(delta_projectionScale, torch.Tensor) else delta_projectionScale


            self.chrome_ngl.change_JSON_state_url(json_state)
        print("Decision acted upon")

    def follow_episode(self, episode):
        """"
        This function takes a recording and follows the actions of the user in the Neuroglancer viewer step by step
        At the moment, the JSON state is fully changed which is not definitive behavior (sort of cheating)
        """
        sequence = episode
        time.sleep(self.sleep_time)
        self.chrome_ngl.change_JSON_state_url(json.dumps(sequence[0]["state"]))

        for i in range(1,len(sequence)):
            start_time = time.time()
            #self.chrome_ngl.get_screenshot("./screenshots/screenshot_" + str(i) + ".png")
            #print("time to get screenshot: ", time.time() - start_time)
            print("Step: ", i)
            step = sequence[i] # state_step is a dictionary containing keys: state, action, time
            step_state = step["state"]
            step_action = step["action"]
            print(step_action)
            step_time = step["time"]
            parsed_action, direct_json_change = parse_action(step_action)
            #print(parsed_action)
            if direct_json_change:
                # time.sleep(0.01)
                json_state = json.dumps(step_state)
                self.chrome_ngl.change_JSON_state_url(json_state)
            else:
                #print("About to do a mouse action: ", parsed_action)
                #time.sleep(0.05)
                self.chrome_ngl.mouse_key_action(parsed_action['x'], parsed_action['y'], parsed_action['click_type'], parsed_action['keys_pressed'])
                #print("Mouse action achieved")

    def reset(self):
      self.action_history = []
      self.chrome_ngl.start_neuroglancer_session()

    def parse_episode(self, episode, save_path=None):
        # Function for parsing the episode data into a format that can be used for pretraining (imitation learning)
        # To call this function, we need to start the session first. Then it will change states and take screenshots
        parsed_data = []
        parsed_images = []

        for i in range(0, len(episode)-1):

            self.chrome_ngl.change_JSON_state_url(json.dumps(episode[i]["state"]))
            # we build the action that leads from the previous state to the current state
            # We need to be careful here, the recording saves the action that led to the state with it, not the action taken in the state
            
        
            next_episode = episode[i+1]
            current_episode = episode[i]
            
            next_state = next_episode["state"]
            current_state = current_episode["state"]
            next_action = next_episode["action"]
            
            output_vector = [
                0, 0, 0,  # left_click, right_click, double_click
                0.0, 0.0,  # x, y (mouse position)
                0, 0, 0,  # key_Shift, key_Ctrl, key_Alt
                0,  # json_change
                0.0, 0.0, 0.0,  # delta_position_x, delta_position_y, delta_position_z
                0.0,  # delta_crossSectionScale
                0.0, 0.0, 0.0, 0.0,  # delta_projectionOrientation_q1, q2, q3, q4
                0.0  # delta_projectionScale
            ]
            # Discrete actions are in order: left_click, right_click, double_click, Shift, Ctrl, Alt, JSON_change
            # Continuous actions are in order: x, y, delta_position_x, delta_position_y, delta_position_z, delta_crossSectionScale, delta_projectionOrientation_q1, q2, q3, q4, delta_projectionScale
            screen_action=False
            if "Double Click" in next_action:
                output_vector[2] = 1
                screen_action=True   
                # double click seems to need a refresh for the neurons to appear, why ?  
                self.chrome_ngl.refresh()
                time.sleep(1)
            elif "Left Click" in next_action:
                output_vector[0] = 1  
                screen_action=True
            elif "Right Click" in next_action:
                output_vector[1] = 1  
                screen_action=True

            if "Relative position" in next_action:
                position_data = next_action.split("Relative position: ")[1].split(" with keys:")[0]
                
                # Extract x and y values
                # First, remove the "x=" and "y=" prefixes to parse the values as integers
                position_parts = position_data.split(", y=")
                x = int(position_parts[0].replace("x=", "").strip())  # Remove "x=" prefix
                y = int(position_parts[1].strip())  # Directly parse y value
                
                output_vector[3] = x / self.values.x_factor
                output_vector[4] = y / self.values.y_factor
            
            if "Shift" in next_action:
                output_vector[5] = 1  # key_Shift
            if "Ctrl" in next_action:
                output_vector[6] = 1  # key_Ctrl
            if "Alt" in next_action:
                output_vector[7] = 1  # key_Alt
            
            if screen_action == False:
                # This means the action was a JSON change
                output_vector[8] = 1
            
            
                delta_pos = [
                    (next_state["position"][0] - current_state["position"][0]) / self.values.delta_x_factor,
                    (next_state["position"][1] - current_state["position"][1]) / self.values.delta_y_factor,
                    (next_state["position"][2] - current_state["position"][2]) / self.values.delta_z_factor
                ]
                output_vector[9], output_vector[10], output_vector[11] = delta_pos
                
                output_vector[12] = (next_state["crossSectionScale"] - current_state["crossSectionScale"]) / self.values.delta_crossSectionScale_factor
                
                delta_orientation = [
                    (next_state["projectionOrientation"][0] - current_state["projectionOrientation"][0]) / self.values.delta_q1_factor,
                    (next_state["projectionOrientation"][1] - current_state["projectionOrientation"][1]) / self.values.delta_q2_factor,
                    (next_state["projectionOrientation"][2] - current_state["projectionOrientation"][2]) / self.values.delta_q3_factor,
                    (next_state["projectionOrientation"][3] - current_state["projectionOrientation"][3]) / self.values.delta_q4_factor
                ]
                output_vector[13], output_vector[14], output_vector[15], output_vector[16] = delta_orientation
                
                output_vector[17] = (current_state["projectionScale"] - current_state["projectionScale"]) / self.values.delta_projectionScale_factor
            
            pos_state, curr_image, json_state = self.prepare_state(image_path=f"{save_path}/screenshots/" + str(i) + ".png")

            parsed_data.append({
                "pos_state": pos_state,  # original state for reference
                "action_vector": tuple(output_vector),  # encoded action vector
                "action": next_action,  # action description (optional)
                "json_state": json_state  # the current JSON state (optional)
            })
            parsed_images.append(curr_image)
            print("Output vector: ", output_vector)
            # Save the parsed data to a file (you can choose the format)
        with open(f"{save_path}/data.json", "w") as f:
            json.dump(parsed_data, f, indent=4)
        
        return parsed_data



if __name__ == "__main__":
    rl_agent = Agent(start_session=True)
    rl_agent.chrome_ngl.start_neuroglancer_session()
    time.sleep(1)
    print("Session started")
    for i in range(9, 10):
        file_path = f"./episodes/episode_{i}.json"
        with open(file_path, "r") as file:
            data = json.load(file)
        save_path = f"./normalized_parsed_episodes/episode_{i}/"

        rl_agent.parse_episode(data, save_path)
        print("Episode completed")
    time.sleep(5)


    

        