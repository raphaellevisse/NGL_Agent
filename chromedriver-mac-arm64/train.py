from Agent import Agent
from ActorCritic import ActorCriticModel
import time
import torch
import os
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ActorCriticModel(state_size=10, action_size=18, device=device, batch_size=128) # 10 for 9 values and the image, 18 for 18 possible actions

agent = Agent(headless=False, start_session=True)
agent.chrome_ngl.start_neuroglancer_session()
#print(agent.chrome_ngl.get_url())
#print('Saving screenshot...')

#agent.chrome_ngl.get_screenshot("./screenshot.png")
num_episodes = 5000
max_steps = 128  
target_update_freq = 10 

for episode in range(num_episodes):
    agent.reset()
    model.memory.clear()
    #pos_state, image, json_state = agent.prepare_state(verbose=True)
    total_reward = 0

    for step in range(max_steps):
        print(f"Episode {episode + 1}/{num_episodes}, Step {step + 1}/{max_steps}")
        # Select action
        #print("Making decision...")
        pos_state, curr_image, json_state = agent.prepare_state()

        discrete_probs, continuous_probs = model.action(pos_state, curr_image)
        #print("Probs are", discrete_probs, continuous_probs)
        output_vector = model.build_output_vector(discrete_probs, continuous_probs)
        agent.apply_actions(output_vector, json_state) # the ouput vector will either do a click or shift the view via the json state

        action_probs = [discrete_probs, continuous_probs]
        next_pos_state, next_image, next_json_state = agent.prepare_state()
        reward = model.reward(next_json_state)

        done = False
        model.store_experience(pos_state, curr_image, action_probs, reward, next_pos_state, next_image, done)
        
        pos_state = next_pos_state
        image = next_image
        total_reward += reward

        
    print("Training model at end of episode...")
    model.train()
    

    if (episode+1) % 5 == 0:
        print("Updating target networks...")
        model.update_target_networks()

    if (episode+1) % 50 == 0:
        model.save_model(f"./checkpoints/train_actor_weights_episode_{episode+1}.pt", f"./checkpoints/train_critic_weights_episode_{episode+1}.pt")
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {model.epsilon}")

time.sleep(100)