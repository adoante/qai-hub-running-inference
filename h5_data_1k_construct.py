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

# Function to create an h5 dataset given a list of image_tensors
def construct_h5_dataset(image_paths, set_num):
	processed_images = []

	for image in image_paths:
		print(f"Preprocessing: {image}")
		image = preprocess_image(image)	
		processed_images.append(image)

	print("Initializing Upload!")
	data = dict(image_tensor = processed_images)
	dataset = hub.upload_dataset(data)
	dataset.download(f"./datasets/dataset_{set_num}")

for i in range(50):
	image_paths = [f"./split_imagenet_folders/imagenet_set_{i + 1}/{img}" for img in listdir(f"./split_imagenet_folders/imagenet_set_{i + 1}")]
	construct_h5_dataset(image_paths, i + 1)