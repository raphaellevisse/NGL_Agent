import os
from PIL import Image
import ast
import time
import io
def c_write_action(output_action):
    """From the cluster, writing on an action log that will be read by the host."""
    #print("Output written to log: ", output_action)
    log_directory = "./proxy/"
    log_file_path = log_directory + "actions.txt"

    os.makedirs(log_directory, exist_ok=True)

    with open(log_file_path, "w") as f:
        f.write(str(output_action))



def c_prepare_state():
    """
        We read from the host_states.txt and we read the image from host_images/
    """
    try_number = 0
    image_success=False
    pos_state_success=False
    pos_state = None
    curr_image = None
    while try_number < 100 and (not image_success or not pos_state_success):
        # Try to open and validate image writing
        print("Try number: ", try_number)
        try:
            if not image_success:
                with open("./proxy/host_images/screenshot_bytes2", 'rb') as f:
                    #print("Image found !")
                    curr_bytes = f.read()
                    open_time = time.time()
                    curr_image = Image.open(io.BytesIO(curr_bytes)).convert("RGB")
                    #print("Image opened in ", time.time()-open_time)
                    curr_image.verify()
                    #print("Image was verified by PIL")
                    image_success = True
            if not pos_state_success:
                with open("./proxy/host_states.txt", "r") as f:
                    pos_state = f.readline()
                pos_state = ast.literal_eval(pos_state)
                #print("Pos state read: ", pos_state)
                pos_state_success = True
        except Exception as e:
            print("Error reading state, retrying...", e)
            try_number += 1
            continue
    print("Did number of tries for state access: ", try_number, " and image read success: ", image_success, " and pos state read success: ", pos_state_success)
    #print("Final pos state is", pos_state)
    return pos_state, curr_image

def c_write_action_timer(output_action):
    """From the cluster, writing on an action log that will be read by the host."""

    print("Output written to log: ", output_action)
    log_directory = "./proxy/"
    log_file_path = log_directory + "actions.txt"

    os.makedirs(log_directory, exist_ok=True)
    opening_time = time.time()
    #check size of the file
    step_size = time.time()
    if os.path.exists(log_file_path):
        size = os.path.getsize(log_file_path)
        print("Size of file: ", size)
        size_time = time.time() - step_size
        print("Time to check size: ", size_time)
    fake_file_path = log_file_path + "_fake"
    name_step = time.time()
    os.path.exists(fake_file_path)
    print("Time to check name: ", time.time() - name_step)

    # rename the file
    renaming_time = time.time()
    os.rename(log_file_path, fake_file_path)
    naming_time = time.time() - renaming_time
    print("Time to rename file: ", naming_time)
    os.rename(fake_file_path, log_file_path)
    with open(log_file_path, "w") as f:
        print("Time to open file: ", time.time() - opening_time)
        write_time = time.time()
        f.write(str(output_action))
        print("Time to write to file: ", time.time() - write_time)

    return size_time, naming_time

# def c_verify_action_effect(self):
#     """From the cluster, verifying the effect of the action."""
#     with open("action_effect.txt", "r") as f:
#         effect = f.read()
#         print("Action effect read from log: ", effect)
#         return effect


if __name__ == "__main__":
    output_action_example = [0.6, 0.5, 0.5, 0.5, 0.5, 0.5]
    start_time = time.time()
    for i in range(100):
        c_write_action_timer(output_action_example)
        #c_prepare_state()