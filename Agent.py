from ChromeNGL import ChromeNGL
from utils import parse_action
import time
import json


class Agent:
    def __init__(self, model=None, start_session: bool = False):
        self.action_history = []
        self.state = None
        self.action = None
        self.reward = None

        self.sleep_time = 1 # time between actions
        if start_session:
            self.chrome_ngl = ChromeNGL()
            self.chrome_ngl.start_session()
        else:
            self.chrome_ngl = None
        
        self.model = model
    
    def prepare_state(self):
        state = self.chrome_ngl.get_JSON_state()
        json_state = json.loads(state)
        # for now the state we give in just the parsed position, crossSectionScale, projectionOrientation, projectionScale
        position = json_state["position"]
        crossSectionScale = json_state["crossSectionScale"]
        projectionOrientation = json_state["projectionOrientation"]
        projectionScale = json_state["projectionScale"]
        pos_state = [position, crossSectionScale, projectionOrientation, projectionScale]
        curr_image = self.chrome_ngl.get_screenshot()
        return pos_state, curr_image, json_state
  
    def decision(self):
        # make a decision based on the current state
        pos_state, curr_image, json_state = self.prepare_state()
        # preprocess the state by making it a vector
        # input = [position, crossSectionScale, projectionOrientation, projectionScale, image]
        # MODEL INPUT = [state, image] (or memory, TBD)

        ## Does not work as everything should be boolean for RL, make boolean increments for each action ?
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
        output_vector = self.model.action(pos_state, curr_image)

        # APPLY ACTIONS
        print(output_vector)
        self.apply_actions(output_vector, json_state)
        return output_vector
        
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
        ) = output_vector
        # fitting output_vector back into action space
        x += 1
        y += 1
        x *= 1000
        y *= 1000
        key_pressed = ""
        if key_Shift:
            key_pressed += "Shift, "
        if key_Ctrl:
            key_pressed += "Ctrl, "
        if key_Alt:
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
            json_state["position"][0] += delta_position_x
            json_state["position"][1] += delta_position_y
            json_state["position"][2] += delta_position_z
            json_state["crossSectionScale"] += delta_crossSectionScale
            json_state["projectionOrientation"][0] += delta_projectionOrientation_q1
            json_state["projectionOrientation"][1] += delta_projectionOrientation_q2
            json_state["projectionOrientation"][2] += delta_projectionOrientation_q3
            json_state["projectionOrientation"][3] += delta_projectionOrientation_q4
            json_state["projectionScale"] += delta_projectionScale

            self.change_JSON_state(json.dumps(json_state))
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


if __name__ == "__main__":
    rl_agent = Agent(start_session=True)
    rl_agent.chrome_ngl.start_neuroglancer_session()
    file_path = "/Users/ri5462/Documents/PNI/RLAgent/episodes/episode_4.json"
    with open(file_path, "r") as file:
        data = json.load(file)
    rl_agent.follow_episode(data)
    print("Episode completed")
    time.sleep(50)


    

        