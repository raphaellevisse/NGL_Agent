print("Starting code", flush=True)
import json
import numpy as np
import torch
import os
# go in parent directory
from PIL import Image

def load_episode_data(num_episodes, episodes_path):
    all_episodes = []
    for idx in range(0, num_episodes):
        episode_path = f"{episodes_path}/episode_{idx}/data_reparsed.json"
        screenshots_path = f"{episodes_path}/episode_{idx}/screenshots"
        images = []
        with open(episode_path, "r") as f:
            episode_data = json.load(f)

        for i in range(len(episode_data)-1):
            current_state = episode_data[i]
            #current_state['screenshot'] = images[i]
            #current_state['next_pos_state'] = episode_data[i+1]['pos_state']
            #current_state['next_screenshot'] = images[i+1]
            all_episodes.append(current_state)
    return all_episodes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)
#agent = Agent(model, start_session=False)

episodes_path = "./reparsed_episodes/"
num_episodes = 13
# This could be done elsewhere but it is sufficiently fast to be done directly here
episodes_data = load_episode_data(num_episodes, episodes_path)
# Initialize variables to accumulate stats
all_actions = []
print(len(episodes_data))
# Iterate through all episodes
for episode in episodes_data:
    action = torch.tensor(episode['action_vector'], dtype=torch.float32).to(device)
    
    # Append to the list for global statistics
    all_actions.append(action)

# Stack all actions into a single tensor
all_actions_tensor = torch.stack(all_actions, dim=0)

# Calculate statistics
max_action = torch.max(all_actions_tensor, dim=0).values
min_action = torch.min(all_actions_tensor, dim=0).values
mean_action = torch.mean(all_actions_tensor, dim=0)
variance_action = torch.var(all_actions_tensor, dim=0)

# Print the results
print(f"Max Action: {max_action}")
print(f"Min Action: {min_action}")
print(f"Mean Action: {mean_action}")
print(f"Variance Action: {variance_action}")


# Max Action: tensor([1.0000e+00, 1.0000e+00, 0.0000e+00, 9.6800e-01, 6.9300e-01, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 1.0000e+00, 4.8212e+03, 1.1563e+03, 4.0000e+00,
#         5.1335e+01, 2.0762e-01, 2.4648e-01, 2.1690e-01, 2.1872e-01, 0.0000e+00])
# Min Action: tensor([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.3635e+03,
#         -1.0524e+03, -4.1279e+00, -5.0272e+01, -2.1178e-01, -2.9298e-01,
#         -2.0002e-01, -2.1217e-01,  0.0000e+00])
# Mean Action: tensor([ 3.0797e-02,  4.5652e-02,  0.0000e+00,  4.6422e-02,  2.8594e-02,
#          0.0000e+00,  0.0000e+00,  0.0000e+00,  9.2355e-01, -3.2228e-01,
#          1.2004e+00,  2.9492e-01,  1.4930e-02, -1.4409e-03,  8.3225e-04,
#         -7.2133e-04,  1.6138e-03,  0.0000e+00])
# Variance Action: tensor([2.9859e-02, 4.3584e-02, 0.0000e+00, 3.0451e-02, 1.1161e-02, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 7.0630e-02, 1.8515e+04, 1.7925e+03, 9.4216e-01,
#         2.3326e+00, 5.8702e-04, 7.5714e-04, 1.0054e-03, 9.2021e-04, 0.0000e+00])