import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torchvision import transforms


### PLACEHOLDER 
class QNetwork(nn.Module):
    def __init__(self, discrete_dim, continuous_dim):
        super(QNetwork, self).__init__()
        discrete_action_dim = discrete_dim
        continuous_action_dim = continuous_dim
        # Image processing layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(32 * 256 * 256 + 9, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.continuous_head = nn.Linear(64, continuous_action_dim)
        self.discrete_head = nn.Linear(64, discrete_action_dim)


    def forward(self, state, image):
        """
        Forward pass through the Q-network.
        :param state: Tensor of shape (batch_size, 9) representing the state vector.
        :param image: Tensor of shape (batch_size, 3, 256, 256) representing the image input.
        :return: Q-values for all possible actions.
        """
        x = self.conv1(image)
        x = self.relu(x)
        x = self.flatten(x)
        
        x = torch.cat((x, state), dim=1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        continuous_actions = torch.tanh(self.continuous_head(x))  # [-1, 1]
        discrete_actions = torch.sigmoid(self.discrete_head(x))  # [0, 1]

        return  discrete_actions, continuous_actions

class RLModel:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        discrete_action_indices = [0, 1, 2, 5, 6, 7, 8]  # Indices for booleans (left_click, right_click, etc.)
        continuous_action_indices = [3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # Indices for floats (x, y, deltas, etc.)
        discrete_dim = len(discrete_action_indices)
        continuous_dim = len(continuous_action_indices)
        self.q_network = QNetwork(discrete_dim, continuous_dim).to(self.device)
        self.target_network = QNetwork(discrete_dim, continuous_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()


        self.discrete_action_indices = discrete_action_indices
        self.continuous_action_indices = continuous_action_indices
    
    def preprocess_state(self, state):
        position, crossSectionScale, projectionOrientation, projectionScale = state
        state_vector = position + [crossSectionScale] + projectionOrientation + [projectionScale]
        return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

    def preprocess_image(self, image):
        transform = transforms.Compose([
                transforms.Resize((256, 256)), 
                transforms.ToTensor(),         
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
            
        # Apply the transformations
        image_tensor = transform(image).unsqueeze(0).to(self.device) 

        return image_tensor

    def action(self, pos_state, image):
        """
        Choose an action based on the current policy (epsilon-greedy).
        """
        # if np.random.rand() <= self.epsilon:
        #     return random.randint(0, self.action_size - 1)  # Random action
        state_tensor = self.preprocess_state(pos_state)
        image_tensor = self.preprocess_image(image)
        print("State tensor and image tensor are", state_tensor.size(), image_tensor.size())
        with torch.no_grad():
            discrete_actions, continuous_actions = self.q_network(state_tensor, image_tensor)
        
        # THESE ARE Q VALUES, NOT ACTIONS --> NEED TO THINK ABOUT RELATIONSHIP TO ACTION
        print("Continous actions and discrete actions are", continuous_actions, discrete_actions)
        output_probs = [discrete_actions, continuous_actions]
        return output_probs
    
    def store_experience(self, pos_state, image, action, reward, next_pos_state, next_image, done):
        """
        Store an experience in replay memory.
        """
        self.memory.append((pos_state, image, action, reward, next_pos_state, next_image, done))

    def train(self):
        """
        Train the Q-network using experiences from the replay memory.
        """
        print("Training the model...")
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, images, actions, rewards, next_states, next_images,dones = zip(*batch)
        
        
        state_tensors = torch.cat([self.preprocess_state(s) for s in states])
        print("State tensors are", state_tensors.size())
        image_tensors = torch.cat([self.preprocess_image(image) for image in images]) 
        next_state_tensors = torch.cat([self.preprocess_state(next_s) for next_s in next_states])
        next_image_tensors = torch.cat([self.preprocess_image(next_image) for next_image in next_images])

        # TO BE DETERMINED -----------------
        # with torch.no_grad():
        #     next_continuous_q_values, next_discrete_q_values = self.target_network(next_state_tensors, next_image_tensors)
        #     max_next_continuous_q_values = torch.max(next_continuous_q_values, dim=1)[0]  # For continuous actions
        #     max_next_discrete_q_values = torch.max(next_discrete_q_values, dim=1)[0]  # For discrete actions
        #     targets = torch.tensor(rewards, dtype=torch.float32).to(self.device) + \
        #             self.gamma * (max_next_continuous_q_values + max_next_discrete_q_values) * \
        #             (1 - torch.tensor(dones, dtype=torch.float32).to(self.device))

        # # Forward pass through the Q-network
        # continuous_q_values, discrete_q_values = self.q_network(state_tensors, image_tensors)
        # if self.is_continuous_action_space: 
        #     continuous_loss = nn.MSELoss()(continuous_q_values, targets)
        #     loss = continuous_loss
        # else:  
        #     discrete_q_values = discrete_q_values.gather(1, torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)).squeeze(1)
        #     discrete_loss = nn.MSELoss()(discrete_q_values, targets)
        #     loss = discrete_loss
        # -----------------------------------

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def build_output_vector(self, discrete_actions, continuous_actions):
        discrete_list = discrete_actions.cpu().tolist()
        continuous_list = continuous_actions.cpu().tolist()

        output_vector = [0] * (len(self.discrete_action_indices) + len(self.continuous_action_indices))

        for idx, value in zip(self.discrete_action_indices, discrete_list):
            output_vector[idx] = round(value) 
        for idx, value in zip(self.continuous_action_indices, continuous_list):
            output_vector[idx] = value
        return output_vector
    
    def update_target_network(self):
        """
        Update the target network with the weights from the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def reward(self, json_state):
        z_position = json_state['position'][2]

        return z_position /1000 # value can be changed, testing purposes

