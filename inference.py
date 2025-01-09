from Agent import Agent
from ActorCritic import ActorCriticModel
import torch
from torchvision import transforms
from Values import Values
from PIL import Image
import io
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCriticModel(state_size=10, action_size=18, device=device) # 10 for 9 values and the image, 18 for 18 possible actions
actor_weights = "checkpoints/actor_weights_final_v2.pt"

if not os.path.exists(actor_weights):
     raise FileNotFoundError(f"Actor weights file not found: {actor_weights}")
model.actor.load_state_dict(torch.load(actor_weights, map_location=device, weights_only=True))

model.actor.eval()

agent = Agent(headless=False, start_session=True)
agent.chrome_ngl.start_neuroglancer_session()

agent.chrome_ngl.change_url("http://localhost:8000/client/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B138657.265625%2C80856.6953125%2C1335.916015625%5D%2C%22crossSectionScale%22:4.45933655284782%2C%22projectionOrientation%22:%5B0.09884308278560638%2C0.9041123986244202%2C-0.4155852496623993%2C0.009988739155232906%5D%2C%22projectionScale%22:12029.259719517953%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22%21720575940623044103%22%2C%22%21720575940612843473%22%2C%22720575940641265549%22%2C%22720575940625693080%22%2C%22720575940645528430%22%2C%22720575940622572010%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D")

max_steps = 128  

for step in range(max_steps):
    print(f"Step {step + 1}/{max_steps}")

    pos_state, curr_image, json_state = agent.prepare_state()

    png_image = io.BytesIO()
    curr_image.save(png_image, format="PNG")
    resize_image = Image.open(png_image)

    width, height = resize_image.size
    resize_image.thumbnail((width//2,height//2))
    
    pos_state_tensor = model.preprocess_state(pos_state)
    curr_image_tensor = model.preprocess_image(resize_image)

    discrete_probs, continuous_probs = model.actor(pos_state_tensor, curr_image_tensor)
    output_vector = model.build_output_vector(discrete_probs, continuous_probs)
    agent.apply_actions(output_vector, json_state) # the output vector will either do a click or shift the view via the json state
















