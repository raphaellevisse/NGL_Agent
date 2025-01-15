from Agent import Agent
from ActorCritic import ActorCriticModel
import time
import torch
import os

#torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ActorCriticModel(state_size=10, action_size=18, device=device, image_width=960, image_height=540) # 10 for 9 values and the image, 18 for 18 possible actions
#actor_weights = "./checkpoints/actor_weights_best_click_v2.pt"
# critic_weights = "weights/critic_weights_best_v5.pt"
# if not os.path.exists(actor_weights):
#      raise FileNotFoundError(f"Actor weights file not found: {actor_weights}")
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

agent = Agent(headless=False, start_session=True)
agent.chrome_ngl.start_neuroglancer_session()
#print(agent.chrome_ngl.get_url())
agent.chrome_ngl.change_url("http://localhost:8000/client/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B138657.265625%2C80856.6953125%2C1335.916015625%5D%2C%22crossSectionScale%22:4.45933655284782%2C%22projectionOrientation%22:%5B0.09884308278560638%2C0.9041123986244202%2C-0.4155852496623993%2C0.009988739155232906%5D%2C%22projectionScale%22:12029.259719517953%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22%21720575940612843473%22%2C%22720575940641265549%22%2C%22720575940625693080%22%2C%22720575940645528430%22%2C%22720575940622572010%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D")
#print('Saving screenshot...')

#agent.chrome_ngl.get_screenshot("./screenshot.png")
num_episodes = 5000
max_steps = 128  
target_update_freq = 10 

for episode in range(num_episodes):
    #agent.reset()
    # Setting the agent to a state it has trained on to see his performance on seen environments.
    agent.chrome_ngl.change_url("http://localhost:8000/client/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B138657.265625%2C80856.6953125%2C1335.916015625%5D%2C%22crossSectionScale%22:4.45933655284782%2C%22projectionOrientation%22:%5B0.09884308278560638%2C0.9041123986244202%2C-0.4155852496623993%2C0.009988739155232906%5D%2C%22projectionScale%22:12029.259719517953%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22%21720575940612843473%22%2C%22720575940641265549%22%2C%22720575940625693080%22%2C%22720575940645528430%22%2C%22720575940622572010%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D")
    #print('Saving screenshot...')
    model.memory.clear()
    #pos_state, image, json_state = agent.prepare_state(verbose=True)
    total_reward = 0

    for step in range(max_steps):
        print(f"Episode {episode + 1}/{num_episodes}, Step {step + 1}/{max_steps}")
        # 1. Interact with the environment and collect data
        pos_state, curr_image, json_state = agent.prepare_state(image_width=960, image_height=540) # looks for the state in the host_states.txt and the image in host_images/host_image.png
        # print("Accessed state !: ", pos_state)
        # print("Accessed image !: ", curr_image)
        # print("Accessed json state !: ", json_state)
        #discrete_probs, continuous_probs = model.action(pos_state, curr_image, eval=True)
        #lets go directly with the actor
        pos_state_tensor = model.preprocess_state(pos_state)
        curr_image_tensor = model.preprocess_image(curr_image)
        #print("Pos state tensor ", pos_state_tensor)
        #print("Curr image tensor ", curr_image_tensor.shape)
        discrete_probs, continuous_probs = model.actor(pos_state_tensor, curr_image_tensor)
        output_vector = model.build_output_vector(discrete_probs, continuous_probs)

        agent.apply_actions(output_vector)
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