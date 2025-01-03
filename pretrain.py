print("Starting code", flush=True)
import json
import numpy as np
import torch
from ActorCritic import ActorCriticModel
from Agent import Agent
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

def load_episode_data(num_episodes, episodes_path):
    all_episodes = []
    for idx in range(0, num_episodes):
        episode_path = f"{episodes_path}/episode_{idx}/data.json"
        screenshots_path = f"{episodes_path}/episode_{idx}/screenshots"
        images = []
        with open(episode_path, "r") as f:
            episode_data = json.load(f)
        for i in range(len(episode_data)):
            screenshot_path = os.path.join(screenshots_path, f"{i}.png")
            if os.path.exists(screenshot_path):
                image = Image.open(screenshot_path)
                width, height = image.size
                new_size = (width // 2, height // 2)
                image.thumbnail(new_size)
                images.append(image)
            else:
                print(f"Screenshot not found at {screenshot_path}")
                images.append(None)

        for i in range(len(episode_data)-1):
            current_state = episode_data[i]
            current_state['screenshot'] = images[i]
            current_state['next_pos_state'] = episode_data[i+1]['pos_state']
            current_state['next_screenshot'] = images[i+1]
            all_episodes.append(current_state)
    return all_episodes

# Pre-training the model using imitation learning with both actor and critic
def pretrain_model(episodes_data, model, num_epochs=10, batch_size=32, gamma=0.99):
    """
    Perform pre-training using imitation learning with the given model and episodes data.
    
    :param episodes_data: The list of episodes data (state, action pairs).
    :param model: The ActorCritic model.
    :param agent: The agent that will use the model.
    :param num_epochs: The number of epochs to train for.
    :param batch_size: The size of each batch for training.
    :param gamma: The discount factor for TD error.
    """
    np.random.shuffle(episodes_data)
    
    for epoch in range(num_epochs):
        total_actor_loss = 0
        total_critic_loss = 0
        
        for i in range(0, len(episodes_data), batch_size):
            batch = episodes_data[i:i+batch_size]
            pos_states = []
            actions = []
            rewards = []
            next_pos_states = []
            images = []
            next_images = []
            for episode in batch:
                json_state = episode['json_state']
                pos_state = model.preprocess_state(episode['pos_state']).to(device)
                action = torch.tensor(episode['action_vector'], dtype=torch.float32).to(device)
                image = model.preprocess_image(episode['screenshot']).to(device)
                next_image = model.preprocess_image(episode['next_screenshot']).to(device)
                reward = torch.tensor(model.reward(json_state), dtype=torch.float32).to(device)
                next_pos_state = model.preprocess_state(episode['next_pos_state']).to(device)

                pos_states.append(pos_state)
                actions.append(action)
                images.append(image)
                next_images.append(next_image)
                rewards.append(reward)
                next_pos_states.append(next_pos_state)
            
            #print("Preparing tensors", flush=True)
            pos_states_tensor = torch.cat(pos_states, dim=0)
            actions_tensor = torch.stack(actions)
            rewards_tensor = torch.stack(rewards)
            next_states_tensor = torch.cat(next_pos_states, dim=0)
            images_tensor = torch.stack(images).squeeze(1)
            next_images_tensor = torch.stack(next_images).squeeze(1)
            # print("Tensors prepared", flush=True)
            # print("Pos states shape", pos_states_tensor.shape, flush=True)
            # print("Actions shape", actions_tensor.shape, flush=True)
            # print("Rewards shape", rewards_tensor.shape, flush=True)
            # print("Next states shape", next_states_tensor.shape, flush=True)
            # print("Images shape", images_tensor.shape, flush=True)
            # print("Next images shape", next_images_tensor.shape, flush=True)

            # Train the Actor (Imitation learning)

            discrete_probs, continuous_probs = model.actor(pos_states_tensor, images_tensor)
            #print("Probs shape",discrete_probs.shape, continuous_probs.shape, flush=True)
            decision_logits= model.build_output_logits(discrete_probs, continuous_probs)
            #print(decision_logits.shape, actions_tensor.shape)
            actor_loss = torch.nn.CrossEntropyLoss()(decision_logits, actions_tensor)
            
            value_estimates = model.critic(pos_states_tensor, images_tensor)
            next_value_estimates = model.critic(next_states_tensor, next_images_tensor)
            
            # Calculate TD error
            td_error = rewards_tensor + gamma * next_value_estimates - value_estimates
            critic_loss = torch.mean(td_error ** 2)  # MSE loss for critic

            total_loss = actor_loss + critic_loss
            
            model.optimizer_actor.zero_grad()
            actor_loss.backward()
            model.optimizer_actor.step()

            # Optimize critic
            model.optimizer_critic.zero_grad()
            critic_loss.backward()
            model.optimizer_critic.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            print(f"Processing batch at {i} out of {len(episodes_data)}", flush=True)

        print(f"Epoch {epoch+1}/{num_epochs}, Actor Loss: {total_actor_loss / len(episodes_data)}, Critic Loss: {total_critic_loss / len(episodes_data)}")
        if (epoch + 1) % 50 == 0:
            model.save_model(f"./checkpoints/actor_weights_epoch_{epoch+1}.pt", f"./checkpoints/critic_weights_epoch_{epoch+1}.pt")


state_size = 10 
action_size = 18 
model = ActorCriticModel(state_size=state_size, action_size=action_size, device=device)
agent = Agent(model, start_session=False)

episodes_path = "./parsed_episodes/"
num_episodes = 13
# This could be done elsewhere but it is sufficiently fast to be done directly here
episodes_data = load_episode_data(num_episodes, episodes_path)
#print("Parsing data")
#episodes_data = torch.load('./pretrain_data.pt')
print(f"Loaded {len(episodes_data)} episodes", flush=True)
pretrain_model(episodes_data, model, batch_size=64, num_epochs=1000)

model.save_model("./checkpoints/actor_weights_final_v1.pt", "./checkpoints/critic_weights_final_v1.pt")