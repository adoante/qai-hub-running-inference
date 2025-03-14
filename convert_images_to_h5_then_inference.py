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
image_paths  = [("./split_imagenet_folders/imagenet_set_1/" + img) for img in listdir("./split_imagenet_folders/imagenet_set_1")]

tensor_image_list = []
for index, image_path in enumerate(image_paths):
	tensor_image_list.append(preprocess_image(image_path))

print(len(tensor_image_list))
data = dict(image_tensor = tensor_image_list)

hub_dataset = hub.upload_dataset(data)

# Submit inference job
inference_job = hub.submit_inference_job(
	model = hub.get_model("mn06pd49n"), # Using previously uploaded model
	device = hub.Device("Samsung Galaxy S24 (Family)"),
	inputs = hub_dataset,
)

# Ensure the job is valid
assert isinstance(inference_job, hub.InferenceJob)

output_data = inference_job.download_output_data(f"./h5_data/wideresnet50_set_1")

# Sets are from 1 - 100
# 000: 00001 - 00100
# 001: 00201 - 00200
# 002: 00301 - 00300
# ...
# 499: 49001 - 50000