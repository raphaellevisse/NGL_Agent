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


# Max Action: tensor([1.0000e+00, 1.0000e+00, 1.0000e+00, 1.4520e+03, 6.9300e+02, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 1.0000e+00, 2.0969e+01, 1.1563e+03, 4.0000e+00,
#         2.3872e+00, 2.0762e-01, 2.4648e-01, 2.1690e-01, 2.1872e-01, 3.7135e+03])
# Min Action: tensor([ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
#          0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, -4.4743e+02,
#         -3.6109e+01, -2.0000e+00,  0.0000e+00, -2.1178e-01, -2.9298e-01,
#         -2.0002e-01, -1.5845e-01, -8.7997e-01])
# Mean Action: tensor([ 9.3371e-04,  5.6489e-02,  2.1008e-02,  7.5114e+01,  2.8972e+01,
#          0.0000e+00,  0.0000e+00,  0.0000e+00,  9.2157e-01, -9.2419e-01,
#          1.4950e+00,  2.2389e-01,  5.1557e-03, -2.3375e-03,  8.3476e-04,
#         -6.9265e-04,  2.4668e-03,  1.7921e+00])
# Variance Action: tensor([9.3327e-04, 5.3323e-02, 2.0577e-02, 7.5191e+04, 1.0959e+04, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 7.2314e-02, 1.8582e+02, 1.0954e+03, 6.4959e-01,
#         5.2019e-03, 6.5645e-04, 8.6409e-04, 1.0306e-03, 9.5943e-04, 6.4426e+03])