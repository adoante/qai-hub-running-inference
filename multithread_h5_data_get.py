import numpy as np
from PIL import Image
import qai_hub as hub
from os import listdir, makedirs
import os
import threading

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))  # Resize to 224x224
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to run inference on a subset of images using a specific device
def process_images(device, image_paths, start_index, output_dir):
    image_num = start_index

    for image_path in image_paths:
        image = preprocess_image(image_path)

        
        inference_job = hub.submit_inference_job(
            model=hub.get_model("mn06pd49n"),  # Using previously uploaded model
            device=device,
            inputs=dict(image_tensor=[image]),  # Pass the processed image tensor
        )

        # Download the output data
        inference_job.download_output_data(f"{output_dir}/wideresnet50_{image_num}.h5")

        print(f"Processed {image_path} - Output saved to {output_dir}/wideresnet50_{image_num}.h5")
        image_num += 100

# Paths to images
image_paths = [f"./split_imagenet_folders/imagenet_set_1/{img}" for img in listdir("./split_imagenet_folders/imagenet_set_1")]

# Ensure output directory exists
output_dir = "./h5_data/imagenet_set_1_data"

# Create 100 devices for parallel processing
devices = []

### ------------------------------------------- ###
### WARNING SET device_num WITH EXTREME CAUTION ###
### ------------------------------------------- ###
#   I submitted 400 inference jobs on accident.   #

device_num = 100

for i in range(device_num):
	devices.append(hub.Device("Samsung Galaxy S24 (Family)"))


# Split the images into device_num x 10:

# List 1:  [1, 101, 201, 301, 401, 501, 601, 701, 801, 901]
# List 2:  [2, 102, 202, 302, 402, 502, 602, 702, 802, 902]
# List 3:  [3, 103, 203, 303, 403, 503, 603, 703, 803, 903]
# List 4:  [4, 104, 204, 304, 404, 504, 604, 704, 804, 904]
# List 5:  [5, 105, 205, 305, 405, 505, 605, 705, 805, 905]
# ...
# List 100: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# 
# Since this is threaded we want the first 100 to inference first
# therefor the list are ordered as shown above

def split_list(lst, num_sublists):
    sublists = []  # Initialize an empty list to hold the sublists
    
    for i in range(num_sublists):  # Loop through the range of num_sublists
        sublist = []  # Initialize an empty sublist for each iteration
        
        # Loop through the list and pick every num_sublists-th element starting at index i
        for j in range(i, len(lst), num_sublists):
            sublist.append(lst[j])  # Add the element to the sublist
        
        sublists.append(sublist)  # Add the sublist to the list of sublists
    
    return sublists

image_chunks = split_list(image_paths, device_num)

# Create threads for each device
threads = []

for i in range(len(devices)):
    # Assign a device and image chunk to each thread
    thread = threading.Thread(target=process_images, args=(devices[i], image_chunks[i], i + 1, output_dir))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All images processed.")
