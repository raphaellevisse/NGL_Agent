import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torchvision import transforms

class ActorNetwork(nn.Module):
    def __init__(self, discrete_dim, continuous_dim, image_width=480, image_height=270):
        super(ActorNetwork, self).__init__()
        self.width = image_width
        self.height = image_height
        # Common layers for both actor and critic
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1920x1080 -> 1920x1080
        self.conv2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 960x540 -> 480x270
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 480x270 -> 240x135
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 240x135 -> 120x68
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 120x68 -> 60x34
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the output size after convolutions
        #conv_output_size = 512 * 60 * 34   # This is from the last convolution layer (512 channels, 120x68 spatial size)
        conv_output_size = self._get_conv_output()
        self.fc1 = nn.Linear(conv_output_size + 9, 128)  # Input size = conv_output_size + state size (9)
        self.fc2 = nn.Linear(128, 64)
        self.continuous_head = nn.Linear(64, continuous_dim)
        self.discrete_head = nn.Linear(64, discrete_dim)

    def _get_conv_output(self):
        # Create a dummy input tensor to pass through the network
        dummy_input = torch.zeros(1, 3, self.height, self.width)
        dummy_output = self.relu(self.conv5(self.relu(self.conv4(self.relu(self.conv3(self.relu(self.conv2(dummy_input))))))))
        return dummy_output.numel()
    
    def forward(self, state, image):
        """
        Forward pass through the actor network.
        :param state: Tensor of shape (batch_size, 9) representing the state vector.
        :param image: Tensor of shape (batch_size, 3, 1920, 1080) representing the image input.
        :return: Action distributions for discrete and continuous actions.
        """
        #print("Image shape", image.shape)
        #x = self.conv1(image)
        #x = self.relu(x)
        #print("X shape", x.shape)
        x = self.conv2(image)
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
        
        continuous_actions = torch.tanh(self.continuous_head(x))  # Continuous actions in the range [-1, 1]
        discrete_actions = torch.sigmoid(self.discrete_head(x))  # Discrete actions in the range [0, 1]

        return discrete_actions, continuous_actions

class CriticNetwork(nn.Module):
    def __init__(self, image_width=480, image_height=270):
        super(CriticNetwork, self).__init__()
        self.width = image_width
        self.height = image_height
        # Common layers for both actor and critic
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1920x1080 -> 1920x1080 (grayscale)
        self.conv2 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)  # 960x540 -> 480x270
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # 480x270 -> 240x135
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # 240x135 -> 120x68
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1) # 120x68 -> 60x34
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Calculate the output size after convolutions
        conv_output_size = self._get_conv_output()
        self.fc1 = nn.Linear(conv_output_size + 9, 128)  # Input size = conv_output_size + state size (9)
        self.fc2 = nn.Linear(128, 64)
        self.value_head = nn.Linear(64, 1)  # Output a single value

    def _get_conv_output(self):
        # Create a dummy input tensor to pass through the network
        dummy_input = torch.zeros(1, 3, self.height, self.width)
        dummy_output = self.relu(self.conv5(self.relu(self.conv4(self.relu(self.conv3(self.relu(self.conv2(dummy_input))))))))
        return dummy_output.numel()
    
    def forward(self, state, image):
        """
        Forward pass through the critic network.
        :param state: Tensor of shape (batch_size, 9) representing the state vector.
        :param image: Tensor of shape (batch_size, 1, 1920, 1080) representing the grayscale image input.
        :return: Value of the state.
        """
        #x = self.conv1(image)
        #x = self.relu(x)
        #print("image shape", image.shape)
        x = self.conv2(image)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.relu(x)

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

        # IMAGES
        self.image_width = 480
        self.image_height = 270
        self.actor = ActorNetwork(discrete_dim, continuous_dim, self.image_width, self.image_height).to(self.device)
        self.critic = CriticNetwork(self.image_width, self.image_height).to(self.device)

        # Target networks
        self.target_actor = ActorNetwork(discrete_dim, continuous_dim).to(self.device)
        self.target_critic = CriticNetwork().to(self.device)

        # Copy weights from the main networks to the target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)

        self.discrete_loss_fn = torch.nn.CrossEntropyLoss()
        self.continuous_loss_fn = torch.nn.MSELoss()

        self.tau = 0.01  # Soft update rate for target networks

        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # For exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = 32
        self.memory = deque(maxlen=2000)

    def preprocess_state(self, state):
        position, crossSectionScale, projectionOrientation, projectionScale = state
        state_vector = position + [crossSectionScale] + projectionOrientation + [projectionScale]
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
        return image_tensor

    def action(self, pos_state, image):
        """
        Choose an action based on the current policy (epsilon-greedy).
        """
        if np.random.rand() <= self.epsilon:
            print("Epsilon search:", self.epsilon)
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
        state_tensors = torch.cat([self.preprocess_state(s) for s in states]).to(self.device)
        image_tensors = torch.cat([self.preprocess_image(img) for img in images]).to(self.device)
        next_state_tensors = torch.cat([self.preprocess_state(ns) for ns in next_states]).to(self.device)
        next_image_tensors = torch.cat([self.preprocess_image(nimg) for nimg in next_images]).to(self.device)

        # Get state values from the critic network
        state_values = self.critic(state_tensors, image_tensors)

        with torch.no_grad():
            next_state_values = self.target_critic(next_state_tensors, next_image_tensors).squeeze(1)
            # print("Next state shape", next_state_values.shape)
            # print("Dones shape", dones.shape)
            # print("Rewards shape", rewards.shape)
            targets = rewards + self.gamma * next_state_values * (1 - dones)

        # Calculate the critic loss (mean squared error between predicted and target state values)
        critic_loss = nn.MSELoss()(state_values, targets)

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
        actions_discrete = (actions_discrete == actions_discrete.max(dim=-1, keepdim=True)[0]).float() #converting it into one type of action (one-hot encoding) 
        actions_continuous = torch.cat([a[1] for a in actions], dim=0).to(self.device)

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
        print("Output vector with discrete decision (arg-max) and continuous decision", output_vector)
        return output_vector
    
    def discrete_continuous_from_actions(self, actions_tensor):
        discrete_actions = actions_tensor[:, self.discrete_action_indices]
        continuous_actions = actions_tensor[:, self.continuous_action_indices]

        return discrete_actions, continuous_actions
    
    def build_output_logits(self, discrete_actions, continuous_actions):
        """
        Handles a batch of inputs and generates a batch of output vectors.

        Args:
            discrete_actions (torch.Tensor): Tensor of shape (batch_size, num_discrete_actions).
            continuous_actions (torch.Tensor): Tensor of shape (batch_size, num_continuous_actions).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, action_size) containing output vectors.
        """
        batch_size = discrete_actions.size(0)
        output_vectors = torch.zeros((batch_size, self.action_size), dtype=torch.float32, device=self.device)

        for batch_idx in range(batch_size):
            discrete_list = discrete_actions[batch_idx]
            continuous_list = continuous_actions[batch_idx]

    
            output_vector = torch.zeros(self.action_size, dtype=torch.float32, device=self.device)

            output_vector[self.discrete_action_indices] = discrete_list
            output_vector[self.continuous_action_indices] = continuous_list

            output_vectors[batch_idx] = output_vector

        return output_vectors

    
    def reward(self, json_state):
        z_position = json_state['position'][2]

        return z_position /1000 # value can be changed, testing purposes

    def update_target_networks(self):
        """
        Soft-update target networks.
        """
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, actor_file, critic_file):
        # Save actor and critic network weights
        torch.save(self.actor.state_dict(), actor_file)
        torch.save(self.critic.state_dict(), critic_file)
        print("Model weights saved!")