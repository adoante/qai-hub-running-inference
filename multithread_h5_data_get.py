import numpy as np
from PIL import Image
import qai_hub as hub
from os import listdir, makedirs
import os
import threading

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
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
        output_data = inference_job.download_output_data(f"{output_dir}/wideresnet50_{image_num}.h5")

        print(f"Processed {image_path} - Output saved to {output_data}")
        image_num += 1

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

device_num = 0

for i in range(device_num):
	devices.append(hub.Device("Samsung Galaxy S24 (Family)"))

# Number of chunks we want (10 chunks)
num_chunks = 10
chunk_size = len(image_paths) // num_chunks  # Each chunk will have 100 elements

# Initialize an empty list to hold the chunks
image_chunks = []

# Loop to create the chunks
for i in range(num_chunks):
    # Calculate the start and end index for the current chunk
    start_index = i * chunk_size
    end_index = start_index + chunk_size
    
    # Append the chunk (sublist) to the chunks list
    chunk = image_paths[start_index:end_index]
    image_chunks.append(chunk)

# Print the resulting chunks
for i, chunk in enumerate(image_chunks):
    print(f"Chunk {i + 1}: {chunk}")

# Create threads for each device
threads = []

for i in range(len(devices)):
    # Assign a device and image chunk to each thread
    thread = threading.Thread(target=process_images, args=(devices[i], image_chunks[i], i * chunk_size, output_dir))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All images processed.")
