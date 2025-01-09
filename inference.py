from Agent import Agent
from ActorCritic import ActorCriticModel
import torch
from PIL import Image
import io
import os
import cv2
import numpy as np
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ActorCriticModel(state_size=10, action_size=18, device=device) # 10 for 9 values and the image, 18 for 18 possible actions
actor_weights = "checkpoints/actor_weights_final_v2.pt"

if not os.path.exists(actor_weights):
     raise FileNotFoundError(f"Actor weights file not found: {actor_weights}")
model.actor.load_state_dict(torch.load(actor_weights, map_location=device, weights_only=True))

model.actor.eval()

agent = Agent(headless=False, start_session=True)
agent.chrome_ngl.start_neuroglancer_session()

agent.chrome_ngl.change_url("http://localhost:8000/client/#!%7B%22dimensions%22:%7B%22x%22:%5B4e-9%2C%22m%22%5D%2C%22y%22:%5B4e-9%2C%22m%22%5D%2C%22z%22:%5B4e-8%2C%22m%22%5D%7D%2C%22position%22:%5B160533.40625%2C80462.75%2C2479.5%5D%2C%22crossSectionScale%22:1.8496565995583267%2C%22projectionOrientation%22:%5B-0.11066838353872299%2C-0.7560726404190063%2C0.10504592210054398%2C0.6364527344703674%5D%2C%22projectionScale%22:31260.083367410043%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://https://bossdb-open-data.s3.amazonaws.com/flywire/fafbv14%22%2C%22tab%22:%22source%22%2C%22name%22:%22Maryland%20%28USA%29-image%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://flywire_v141_m783%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%22720575940623044103%22%5D%2C%22name%22:%22flywire_v141_m783%22%7D%5D%2C%22showDefaultAnnotations%22:false%2C%22selectedLayer%22:%7B%22size%22:350%2C%22visible%22:true%2C%22layer%22:%22flywire_v141_m783%22%7D%2C%22layout%22:%22xy-3d%22%7D")

# steps for episode
max_steps = 10  

# video recording
if '-r' in sys.argv:
    output_file = "./videos/recording.mp4"
    video_data = []
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, 5, (480, 270)) #5=frame rate

for step in range(max_steps):
    print(f"Step {step + 1}/{max_steps}")

    pos_state, curr_image, json_state = agent.prepare_state()

    # for video recording
    if '-r' in sys.argv:
        png_image = io.BytesIO()
        curr_image.save(png_image, format="PNG")
        png_image.seek(0)
        video_data.append(png_image.read())
    
    pos_state_tensor = model.preprocess_state(pos_state)
    curr_image_tensor = model.preprocess_image(curr_image)

    discrete_probs, continuous_probs = model.actor(pos_state_tensor, curr_image_tensor)
    output_vector = model.build_output_vector(discrete_probs, continuous_probs)
    agent.apply_actions(output_vector, json_state) # the output vector will either do a click or shift the view via the json state

# image processing
if '-r' in sys.argv:
    for idx, png_data in enumerate(video_data):
        # Load the PNG image using PIL
        img = Image.open(io.BytesIO(png_data))

        # Convert PIL image to NumPy array
        frame = np.array(img)

        # Convert RGB to BGR (OpenCV uses BGR format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add text
        #text = f"Step {idx+1}/{max_steps}"
        #position = (50, 50)  # x, y position of the text
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #font_scale = 1
        #color = (0, 255, 0)  # Green color in BGR
        #thickness = 2
        #cv2.putText(frame, text, position, font, font_scale, color, thickness)

        # Write the frame to the video
        out.write(frame)
    out.release()
    print(f"Video saved to {output_file}")













