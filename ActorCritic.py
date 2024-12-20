import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torchvision import transforms

class ActorNetwork(nn.Module):
    def __init__(self, discrete_dim, continuous_dim):
        super(ActorNetwork, self).__init__()
        
        # Common layers for both actor and critic
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1920x1080 -> 1920x1080
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 1920x1080 -> 960x540
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 960x540 -> 480x270
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 480x270 -> 240x135
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 240x135 -> 120x68
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the output size after convolutions
        conv_output_size = 512 * 120 * 68  # This is from the last convolution layer (512 channels, 120x68 spatial size)
        
        self.fc1 = nn.Linear(conv_output_size + 9, 128)  # Input size = conv_output_size + state size (9)
        self.fc2 = nn.Linear(128, 64)
        self.continuous_head = nn.Linear(64, continuous_dim)
        self.discrete_head = nn.Linear(64, discrete_dim)

    def forward(self, state, image):
        """
        Forward pass through the actor network.
        :param state: Tensor of shape (batch_size, 9) representing the state vector.
        :param image: Tensor of shape (batch_size, 3, 1920, 1080) representing the image input.
        :return: Action distributions for discrete and continuous actions.
        """
        #print("Image shape", image.shape)
        x = self.conv1(image)
        x = self.relu(x)
        #print("X shape", x.shape)
        x = self.conv2(x)
        x = self.relu(x)
        #print("X shape", x.shape)
        x = self.conv3(x)
        x = self.relu(x)
        #print("X shape", x.shape)
        x = self.conv4(x)
        x = self.relu(x)
        #print("X shape", x.shape)
        x = self.conv5(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        #print("X shape", x.shape)
        # Concatenate state vector with image features
        x = torch.cat((x, state), dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        continuous_actions = torch.sigmoid(self.continuous_head(x))  # Continuous actions in the range [-1, 1]
        discrete_actions = torch.sigmoid(self.discrete_head(x))  # Discrete actions in the range [0, 1]

        return discrete_actions, continuous_actions


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        # Common layers for both actor and critic
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1920x1080 -> 1920x1080 (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 1920x1080 -> 960x540
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 960x540 -> 480x270
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 480x270 -> 240x135
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 240x135 -> 120x68
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the output size after convolutions
        conv_output_size = 512 * 120 * 68  # This is from the last convolution layer (512 channels, 120x68 spatial size)
        
        self.fc1 = nn.Linear(conv_output_size + 9, 128)  # Input size = conv_output_size + state size (9)
        self.fc2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)  # Output a single value

    def forward(self, state, image):
        """
        Forward pass through the critic network.
        :param state: Tensor of shape (batch_size, 9) representing the state vector.
        :param image: Tensor of shape (batch_size, 1, 1920, 1080) representing the grayscale image input.
        :return: Value of the state.
        """
        x = self.conv1(image)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        x = self.flatten(x)

        # Concatenate state vector with image features
        x = torch.cat((x, state), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        state_value = self.value_head(x)
        return state_value


class ActorCriticModel:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.discrete_action_indices = [0, 1, 2, 5, 6, 7, 8]  # Indices for booleans (left_click, right_click, etc.)
        self.continuous_action_indices = [3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # Indices for floats (x, y, deltas, etc.)
        discrete_dim = len(self.discrete_action_indices)
        continuous_dim = len(self.continuous_action_indices)

        self.actor = ActorNetwork(discrete_dim, continuous_dim).to(self.device)
        self.critic = CriticNetwork().to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)

        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # For exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 16
        self.memory = deque(maxlen=2000)

    def preprocess_state(self, state):
        position, crossSectionScale, projectionOrientation, projectionScale = state
        state_vector = position + [crossSectionScale] + projectionOrientation + [projectionScale]
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (1 channel)
            transforms.Resize((1920, 1080)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale (1 channel)
        ])
        
        # Apply the transformations
        image_tensor = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
        return image_tensor

    def action(self, pos_state, image):
        """
        Choose an action based on the current policy (epsilon-greedy).
        """
        if np.random.rand() <= self.epsilon:
            discrete_actions = torch.zeros(1, len(self.discrete_action_indices)).to(self.device)
            random_action_index = random.randint(0, discrete_actions.shape[1] - 1)
            discrete_actions[0, random_action_index] = 1
            continuous_actions = torch.rand(1, len(self.continuous_action_indices)).to(self.device)
            return discrete_actions, continuous_actions
        
        state_tensor = self.preprocess_state(pos_state)
        image_tensor = self.preprocess_image(image)

        with torch.no_grad():
            discrete_actions, continuous_actions = self.actor(state_tensor, image_tensor)


        return discrete_actions, continuous_actions

    def store_experience(self, pos_state, image, action, reward, next_pos_state, next_image, done):
        """
        Store an experience in replay memory.
        """
        self.memory.append((pos_state, image, action, reward, next_pos_state, next_image, done))

    def train(self):
        """
        Train both the actor and critic networks using experiences from the replay memory.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, images, actions, rewards, next_states, next_images, dones = zip(*batch)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        state_tensors = torch.cat([self.preprocess_state(s) for s in states])
        image_tensors = torch.cat([self.preprocess_image(img) for img in images])
        next_state_tensors = torch.cat([self.preprocess_state(ns) for ns in next_states])
        next_image_tensors = torch.cat([self.preprocess_image(nimg) for nimg in next_images])

        # Get state values from the critic network
        state_values = self.critic(state_tensors, image_tensors)

        # Calculate the target value
        with torch.no_grad():
            next_state_values = self.critic(next_state_tensors, next_image_tensors).squeeze(1)
            # print("Next state shape", next_state_values.shape)
            # print("Dones shape", dones.shape)
            # print("Rewards shape", rewards.shape)
            targets = rewards + self.gamma * next_state_values * (1 - dones)

        # Calculate the critic loss (mean squared error between predicted and target state values)
        critic_loss = nn.MSELoss()(state_values, targets)

        # Now update the critic network
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Compute advantages
        advantages = targets - state_values.detach()

        # Get action distributions from the actor network
        # Compute advantages
        advantages = (targets - state_values.detach()).squeeze(1)  # Ensure proper shape

        # Get action distributions and continuous outputs from the actor
        predicted_discrete_actions, predicted_continuous_actions = self.actor(state_tensors, image_tensors)
        
        #print("actions is of shape", actions.shape)
        #print("Actions is of shape", actions)
        # For discrete actions, we will get the index of the action with the highest probability
        # For discrete actions, concatenate the action probabilities (or logits) from the batch
        actions_discrete = torch.cat([a[0] for a in actions], dim=0).to(self.device)

        # Convert the action probabilities/logits to a one-hot encoded vector
        actions_discrete = (actions_discrete == actions_discrete.max(dim=-1, keepdim=True)[0]).float()

        #print("Actions discrete", actions_discrete)
        # For continuous actions, we will concatenate the continuous action tensors
        actions_continuous = torch.cat([a[1] for a in actions], dim=0).to(self.device)
        #print("Actions continuous", actions_continuous)

        # Get predicted actions from the actor
        predicted_discrete_actions, predicted_continuous_actions = self.actor(state_tensors, image_tensors)

        # Calculate discrete loss
        log_probs = torch.log(predicted_discrete_actions + 1e-10)  # Avoid log(0)
        discrete_loss = -torch.sum(log_probs * actions_discrete, dim=1)  # Weighted by actual actions
        discrete_loss = torch.mean(discrete_loss * advantages)  # Weighted by advantages

        # Calculate continuous loss
        continuous_loss = nn.MSELoss()(predicted_continuous_actions, actions_continuous)

        # Total loss for actor
        actor_loss = discrete_loss + continuous_loss

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def build_output_vector(self, discrete_actions, continuous_actions):
        # set the maximum value to 1 and the rest to 0
       
        discrete_actions = (discrete_actions == discrete_actions.max()).float()
        discrete_list = discrete_actions.cpu()[0]
        continuous_list = continuous_actions.cpu()[0]

        output_vector = torch.tensor([0.0] * self.action_size, dtype=torch.float32).to(self.device)
        #print(len(output_vector))
        #print(discrete_list)
        #print(continuous_list)
        for i, idx in enumerate(self.discrete_action_indices):
            value = discrete_list[i]
            
            output_vector[idx] = value
        for idx, value in zip(self.continuous_action_indices, continuous_list):
            output_vector[idx] = value
        print("Output vector", output_vector)
        return output_vector

    
    def reward(self, json_state):
        z_position = json_state['position'][2]

        return z_position /1000 # value can be changed, testing purposes
