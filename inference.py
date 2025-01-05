import torch
from torchvision import transforms
from Agent import Agent
from ActorCritic import ActorNetwork

# Start agent
#rl_agent = Agent(start_session=True)
#rl_agent.chrome_ngl.start_neuroglancer_session()

# Load model with weights
model = ActorNetwork(7, 11, 480, 270)
model.load_state_dict(torch.load("./checkpoints/actor_weights_final_v1.pt"))
model.eval()

def preprocess_state(self, state):
    position, crossSectionScale, projectionOrientation, projectionScale = state
    state_vector = position + [crossSectionScale] + projectionOrientation + [projectionScale]
    return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

def preprocess_image(self, image):
    transform = transforms.Compose([
        transforms.Resize((270, 480)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale (1 channel)
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(self.device)  # Add batch dimension
    return image_tensor

def processEnvironment():
    # collect JSON state and image, base function of pretrain.py and load_episode_data()
    pass

# Loop through environmental input and next action prediction until full epoch has been run through
state_tensor = preprocess_state(pos_state)
image_tensor = preprocess_image(image)

predicted_discrete_actions, predicted_continuous_actions = model(state_tensor, image_tensor)