from Agent import Agent
from RLModel import RLModel

# PLACEHOLDER FOR TRAINING
model = RLModel(state_size=10, action_size=18) # 10 for 9 values and the image, 18 for 18 possible actions

agent = Agent(model, start_session=True)
num_episodes = 500 
max_steps = 100     
target_update_freq = 10 

for episode in range(num_episodes):
    agent.reset()
    pos_state, image, json_state = agent.prepare_state()
    total_reward = 0

    for step in range(max_steps):
        # Select action
        print("Making decision...")
        action = agent.decision() # this calls the agent to get the environment
        print("Decision Made")
        next_pos_state, next_image, next_json_state = agent.prepare_state()
        reward = model.reward(next_json_state)

        done = False
        model.store_experience(pos_state, image, action, reward, next_pos_state, next_image, done)
        
        
        model.train() # Does not work yet, have to think about the action state more
        
        pos_state = next_pos_state
        image = next_image
        total_reward += reward
        
    
    if episode % target_update_freq == 0:
        model.update_target_network()


    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {model.epsilon}")
