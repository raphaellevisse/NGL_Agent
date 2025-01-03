from Agent import Agent
from RLModel import RLModel
from ActorCritic import ActorCriticModel
import time
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ActorCriticModel(state_size=10, action_size=18, device=device, batch_size=32) # 10 for 9 values and the image, 18 for 18 possible actions

agent = Agent(headless=True, start_session=True)
agent.chrome_ngl.start_neuroglancer_session()
#print(agent.chrome_ngl.get_url())
#print('Saving screenshot...')

#agent.chrome_ngl.get_screenshot("./screenshot.png")
num_episodes = 500 
max_steps = 100     
target_update_freq = 10 

for episode in range(num_episodes):
    agent.reset()
    #pos_state, image, json_state = agent.prepare_state(verbose=True)
    total_reward = 0

    for step in range(max_steps):
        # Select action
        print("Making decision...")
        pos_state, curr_image, json_state = agent.prepare_state()

        discrete_probs, continuous_probs = model.action(pos_state, curr_image)
        print("Probs are", discrete_probs, continuous_probs)
        output_vector = model.build_output_vector(discrete_probs, continuous_probs)
        agent.apply_actions(output_vector, json_state) # the ouput vector will either do a click or shift the view via the json state

        action_probs = [discrete_probs, continuous_probs]
        next_pos_state, next_image, next_json_state = agent.prepare_state()
        reward = model.reward(next_json_state)

        done = False
        model.store_experience(pos_state, curr_image, action_probs, reward, next_pos_state, next_image, done)
        
        
        model.train() 
        
        pos_state = next_pos_state
        image = next_image
        total_reward += reward
        
    
    if episode % target_update_freq == 0:
        model.update_target_networks()

    model.save_model(f"./checkpoints/actor_weights_episode_{episode+1}.pt", f"./checkpoints/critic_weights_episode_{episode+1}.pt")
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {model.epsilon}")

time.sleep(100)