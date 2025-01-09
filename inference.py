from Agent import Agent
from ActorCritic import ActorNetwork
import torch
from torchvision import transforms
from Values import Values
from PIL import Image
import io


# initialize pytorch
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# normalization values & other values
values = Values()
action_size = 18
continuous_action_indices = [3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17]
discrete_action_indices = [0, 1, 2, 5, 6, 7, 8]

# Load model with weights
model = ActorNetwork(discrete_dim=7, continuous_dim=11, image_width=480, image_height=270)
model.load_state_dict(torch.load("./checkpoints/actor_weights_final_v2.pt"))
model.eval()



# preprocess inputs
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ])
    
    image_tensor = transform(image).unsqueeze(0).to("cpu")  # Add batch dimension
    return image_tensor
def preprocess_state(state):
    position, crossSectionScale, projectionOrientation, projectionScale = state
    norm_position = [
    position[0] / values.position_x_factor,
    position[1] / values.position_y_factor,
    position[2] / values.position_z_factor,
    ]
    norm_crossSectionScale = crossSectionScale /(values.crossSectionScale_factor)
    #print(values.crossSectionScale_factor)

    norm_projectionOrientation = [
    projectionOrientation[0] / values.projectionOrientation_q1_factor,
    projectionOrientation[1] / values.projectionOrientation_q2_factor,
    projectionOrientation[2] / values.projectionOrientation_q3_factor,
    projectionOrientation[3] / values.projectionOrientation_q4_factor,
    ]
    norm_projectionScale = projectionScale / values.projectionScale_factor

    print("projection section scale"+ str(projectionScale))
    state_vector = norm_position + [norm_crossSectionScale] + norm_projectionOrientation + [norm_projectionScale]
    return torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).to("cpu")

# build output vector
def build_output_vector(discrete_actions, continuous_actions):
    # set the maximum value to 1 and the rest to 0
    print("action probabilities")
    
    discrete_actions = (discrete_actions == discrete_actions.max()).float()
    discrete_list = discrete_actions.cpu()[0]
    continuous_list = continuous_actions.cpu()[0]

    output_vector = torch.tensor([0.0] * action_size, dtype=torch.float32).to("cpu")
    #print(len(output_vector))
    
    for i, idx in enumerate(discrete_action_indices):
        value = discrete_list[i]
        
        output_vector[idx] = value
    for idx, value in zip(continuous_action_indices, continuous_list):
        output_vector[idx] = value
    print("Output vector with discrete decision (arg-max) and continuous decision", output_vector)
    return output_vector



agent = Agent(headless=False, start_session=True)
agent.chrome_ngl.start_neuroglancer_session()
#print(agent.chrome_ngl.get_url())
#print('Saving screenshot...')

#agent.chrome_ngl.get_screenshot("./screenshot.png")
num_episodes = 1
max_steps = 200
#target_update_freq = 10 

for episode in range(num_episodes):
    agent.reset()
    #model.memory.clear()
    #pos_state, image, json_state = agent.prepare_state(verbose=True)
    #total_reward = 0

    for step in range(max_steps):
        print(f"Episode {episode + 1}/{num_episodes}, Step {step + 1}/{max_steps}")
        # Select action
        #print("Making decision...")
        pos_state, curr_image, json_state = agent.prepare_state()

        png_image = io.BytesIO()
        curr_image.save(png_image, format="PNG")
        resize_image = Image.open(png_image)

        width, height = resize_image.size
        resize_image.thumbnail((width//2,height//2))

        state_tensor = preprocess_state(pos_state)
        image_tensor = preprocess_image(resize_image)

        print("raw inputs")
        print(state_tensor)
        print(pos_state)

        with torch.no_grad():
            discrete_probs, continuous_probs = model(state_tensor, image_tensor)

        output_vector = build_output_vector(discrete_probs, continuous_probs)
        agent.apply_actions(output_vector, json_state)
