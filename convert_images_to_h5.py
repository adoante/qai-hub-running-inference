import numpy as np
from PIL import Image
import qai_hub as hub
from os import listdir

def preprocess_image(image_path):
	print(f"Preprocessing: {image_path}")

	# Open the image using PIL
	image = Image.open(image_path)

	# Resize the image to 224x224 (expected by WideResNet50)
	image = image.resize((224, 224))

	# Convert the image to numpy array and normalize if required (e.g., scaling to [0, 1] range)
	image = np.array(image).astype(np.float32) / 255.0

	# The model expects a batch dimension, so we add one (shape becomes (1, 224, 224, 3))
	image = np.expand_dims(image, axis=0)

	return image

# Paths to images
image_paths  = [("./ILSVRC2012_img_val/" + img) for img in listdir("./ILSVRC2012_img_val")]

# Define the number of chunks
num_chunks = 500
chunk_size = len(image_paths) // num_chunks  # Each chunk will have 100 elements

# Split the list
split_image_paths = []

# split_image_paths -> 100 x 500 2d array
for i in range(0, len(image_paths), chunk_size):
	split_image_paths.append(image_paths[i:i + chunk_size])

for index, split_image_path in enumerate(split_image_paths):
	data = dict()
	tensor_image_list = []

	for image in split_image_path:
		tensor_image_list.append(preprocess_image(image))

	data = dict(
 	      image_tensor = tensor_image_list
 	)

	print(f"Uploading dataset: {index}")
	hub_dataset = hub.upload_dataset(data)

	hub_dataset.download(f"./datasets/imagenet1k_set_{index}")

# Sets are from 1 - 100
# 000: 00001 - 00100
# 001: 00201 - 00200
# 002: 00301 - 00300
# ...
# 499: 49001 - 50000