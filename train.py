from Agent import Agent
from ActorCritic import ActorCriticModel
import time
import torch
import os
from proxy.cluster_help import c_write_action, c_prepare_state


#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ActorCriticModel(state_size=10, action_size=18, device=device) # 10 for 9 values and the image, 18 for 18 possible actions
#actor_weights = "./checkpoints/actor_weights_best_click_v2.pt"
# critic_weights = "weights/critic_weights_best_v5.pt"
#if not os.path.exists(actor_weights):
#     raise FileNotFoundError(f"Actor weights file not found: {actor_weights}")
# if not os.path.exists(critic_weights):
#     raise FileNotFoundError(f"Critic weights file not found: {critic_weights}")
# if os.path.exists(actor_weights) and os.path.exists(critic_weights):
#     model.load_model(actor_weights, critic_weights)
#     print("Model loaded successfully.")
#model.actor.load_state_dict(torch.load(actor_weights, map_location=device, weights_only=True))
#model.critic.load_state_dict(torch.load(critic_weights, map_location=device, weights_only=True))
model.actor.eval()
model.critic.eval()
model.target_actor.eval()
model.target_critic.eval()

# --- host agent needs to be already initialized ----

num_episodes = 5000
max_steps = 128  
target_update_freq = 10 

for episode in range(num_episodes):
    #agent.reset()
    model.memory.clear()
    total_reward = 0

    
    for step in range(max_steps):
        start_time = time.time()
        print(f"Episode {episode + 1}/{num_episodes}, Step {step + 1}/{max_steps}", flush=True)
        prep_time = time.time()
        pos_state, curr_image = c_prepare_state() # looks for the state in the host_states.txt and the image in host_images/host_image.png
        print("Accessing data took: ", time.time() - prep_time, flush=True)
        #print("Accessed state !: ", pos_state)
        #print("Pos is of shape ", len(pos_state))
        #print("Accessed image !: ", curr_image)
        #discrete_probs, continuous_probs = model.action(pos_state, curr_image, eval=True)
        #lets go directly with the actor
        inference_time= time.time()
        pos_state_tensor = model.preprocess_state(pos_state)
        curr_image_tensor = model.preprocess_image(curr_image)
        #print("Pos state tensor ", pos_state_tensor)
        #print("Curr image tensor ", curr_image_tensor.shape)
        discrete_probs, continuous_probs = model.actor(pos_state_tensor, curr_image_tensor)
        output_vector = model.build_output_vector(discrete_probs, continuous_probs)
        print("Inference took: ", time.time() - inference_time, flush=True)
        tensor_str = str(output_vector.tolist())
        #print("Tensor string: ", tensor_str)
        write_time = time.time()
        c_write_action(tensor_str)
        print("Writing action took: ", time.time() - write_time, flush=True) 

        # HOST SEES ACTION -> ACTS
        #time.sleep(0.1) # wait for the host to act
        print("Episode time took: ", time.time() - start_time, flush=True)
        #c_verify_action_effect()

        # next_pos_state, next_image, next_json_state = agent.prepare_state()
        # reward = model.reward(next_json_state)

        # done = False
        # model.store_experience(pos_state, curr_image, discrete_probs, continuous_probs, reward, next_pos_state, next_image, done)
        
        # pos_state = next_pos_state
        # image = next_image
        # total_reward += reward
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Epsilon: {model.epsilon}")
    continue
    print("Training model at end of episode...")
    
    model.train()
    

    if (episode+1) % 5 == 0:
        print("Updating target networks...")
        model.update_target_networks()

    if (episode+1) % 50 == 0:
        model.save_model(f"./checkpoints/train_actor_weights_episode_{episode+1}.pt", f"./checkpoints/train_critic_weights_episode_{episode+1}.pt")
    

time.sleep(100)