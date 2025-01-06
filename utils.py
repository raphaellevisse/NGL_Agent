import re

def parse_action(action: str):
    json_keywords = ["Outside render:", "Drag", "Wheel", "Keyboard"]
    if any(keyword in action for keyword in json_keywords):
        # any of these operations act directly on the JSON state, the agent will learn to operate directly on it
        return None, True
    
    parsed_result = {
            'x': None,
            'y': None,
            'click_type': None,
            'keys_pressed': None,
            'json_change': False
        }
        
    
    position_pattern = r"Relative position: x=(\d+), y=(\d+)"
    double_click_pattern = r"Double Click"
    single_click_pattern = r"Single Click: ([\w\s]+) Click"
    keys_pattern = r"with keys: ([\w\s,]+)"
    
    
    position_match = re.search(position_pattern, action)
    if position_match:
        parsed_result['x'] = int(position_match.group(1))
        parsed_result['y'] = int(position_match.group(2))
    
    
    if re.search(double_click_pattern, action):
        parsed_result['click_type'] = "double_click"
    else:
        single_click_match = re.search(single_click_pattern, action)
        if single_click_match:
            parsed_result['click_type'] = single_click_match.group(1).strip().lower()
    
    keys_match = re.search(keys_pattern, action)
    if keys_match:
        parsed_result['keys_pressed'] = keys_match.group(1).strip()
    return parsed_result, False

if __name__ == "__main__":
    action_string = "Action Event: Inside render: Single Click: Right Click | Relative position: x=1521, y=40 with keys: Ctrl, Alt"
    parsed = parse_action(action_string)
    print(parsed)




      