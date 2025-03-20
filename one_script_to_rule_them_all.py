import qai_hub as hub
import numpy as np
import pathlib
import re
import h5py
import json

from PIL import Image
from enum import Enum

# Defines the only two input spec
class InputSpec(Enum):
	NORMAL = np.float32
	QUANTIZED = np.uint8

class FileType(Enum):
	TFLITE = False
	ONNX = True

# Preprocess image based on AI model input spec
def preprocess_image(image_path, input_spec, file_type):
	# Open image and account for grey scale
	image = Image.open(image_path).convert("RGB")

	# All image classification models require a size of 224x224
	image = image.resize((224, 224))

	# Convert to np.array with specific type
	image = np.array(image).astype(input_spec.value)

	# Normalize only if the input spec is NORMAL
	if input_spec is InputSpec.NORMAL:
		image = image / 255.0  # Normalize to [0, 1]
	
	# Rearrange dimensions: (H, W, C) → (C, H, W)
	if file_type is FileType.ONNX:
		image = np.transpose(image, (2, 0, 1))  # Shape: (3, 224, 224)

	# All image classification models require a batch dimension of 1
	image = np.expand_dims(image, axis=0)

	return image

# Create an h5 dataset and download
def construct_datasets(image_paths, datasets_dir, dataset_num, input_spec, file_type):
	# Process each image given
	processed_images = []

	for image in image_paths:
		print(f"🖼️ Preprocessing: {image[-28:]} 🖼️")
		image = preprocess_image(image, input_spec, file_type)	
		processed_images.append(image)

	# Upload dataset to QAI Hub to create h5 file
	print("🔥 Initializing Upload! 🔥")
	data = dict(image_tensor = processed_images)
	dataset = hub.upload_dataset(data)

	# Download dataset from QAI Hub into folder
	datasets_dir = pathlib.Path(datasets_dir)
	datasets_dir.mkdir(parents=True, exist_ok=True)

	download_path = f"./{datasets_dir}/dataset_{dataset_num}"
	if input_spec is InputSpec.QUANTIZED:
		download_path = f"./{datasets_dir}/dataset_{input_spec.name.lower()}_{dataset_num}"
	
	dataset.download(download_path)

# Run inference on  QAI Hub and download
def inference_dataset(dataset_paths, model_id, device_name, model_name, results_dir):
	for dataset in dataset_paths:
		# Grab the dataset id
		match = re.search(r'_(\d+)\.h5', dataset)
		if not match:
			print("❌ Dataset naming incorrect!")
			print(f"❌ Got: {dataset}")
			print("❌ Expected: some/path/dataset_'DATASET_ID'.h5")
			print("❌ Expected: some/path/dataset_quantized_'DATASET_ID'.h5")
			exit()
		dataset_id = int(match.group(1))
		
		# Submit inference job
		print(f"😩 Inference: {dataset} 😩")
		inference_job = hub.submit_inference_job(
			model = hub.get_model(model_id),
			device = hub.Device(device_name),
			inputs = dataset,
		)

		# Ensure the job is valid
		assert isinstance(inference_job, hub.InferenceJob)

		# Ensure folder exists
		results_dir = pathlib.Path(results_dir)
		results_dir.mkdir(parents=True, exist_ok=True)

		# Download inference results dataset to specific folder
		download_path = f"./{results_dir}/{model_name}_{dataset_id}"
		inference_job.download_output_data(download_path)

# Results are stored as logits in h5 file and save as json
def process_results(result_paths, class_index_path, synset_path):
	data = dict()
	data_key = 1

	# Load ImageNet class indexes
	with open(class_index_path, "r") as file:
		class_indexes = json.load(file)

	# Load Synset JSON from the file
	with open(synset_path, "r") as file:
		synset_dict = json.load(file)

	for result in result_paths:
		print(f"👩‍🔬Processing result: {result} 👩‍🔬")

		with h5py.File(result, "r") as result_file:
			batches = result_file["/data"]["0"]
			result_batch_paths = [
				"/data/0/batch_" + str(batch_num) for batch_num in range(len(batches))
			]

			for result_batch in result_batch_paths:
				logits = result_file[result_batch][()]  
				
				# Apply softmax to logits
				probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

				# Top 5 results
				top5_results = []

				# Top 5 Predictions
				top5_predictions = np.argsort(probabilities[0], axis=0)[-5:][::-1]

				# Construct Top 5 results list
				for class_index in top5_predictions:
					wnid = class_indexes[str(class_index)][0]  # Find synset id
					synset = synset_dict[wnid]
					top5_results.append(str(synset))

				# Add Top 5 results to output
				data[f"{data_key}"] = top5_results
				data_key += 1
	
	# Save results
	with open("results.json", "w") as results_file:
		json.dump(data, results_file)

def calculate_accuracy(results_json, ground_truth_json, device_name, model_name, library_name):
	with open(ground_truth_json, "r") as file:
		ground_truth = json.load(file)
	
	with open(results_json, "r") as file:
		results = json.load(file)

	# Initialize counters for accuracy
	correct_top1 = 0
	correct_top5 = 0
	total_results = len(results)

	# Calculate top 1
	for i in range(0, total_results):
		# Get top 1 and top 5 predictions
		top5_predictions = results[str(i + 1)]
		top1_prediction = top5_predictions[0]
		
		# Get ground truth
		gt = ground_truth[str(i + 1)]
		
		# Update accuracy counters
		if top1_prediction == gt:
			correct_top1 += 1

		if gt in top5_predictions:
			correct_top5 += 1

	# Calculate top 1 and top 5 accuracy
	top1_accuracy = round((correct_top1 / total_results) * 100, 2)
	top5_accuracy = round((correct_top5 / total_results) * 100, 2)

	accuracy_results = (top1_accuracy, top5_accuracy, device_name, model_name, library_name)

	# Save results to JSON file
	with open("model_accuracy_scores.txt", "a") as accuracy_file:
		accuracy_file.write(str(accuracy_results) + "\n")

	print(str(accuracy_results) + " 😀")

###
### I still don't get this
###

# Sorting the filenames based on the numerical part after '_'
def extract_number(filename):
	parts = filename.split('_')  # Split by '_'

	if len(parts) > 1:  # Ensure there is a number part
		num_part = ''.join(filter(str.isdigit, parts[-1]))  # Extract digits
		return int(num_part) if num_part.isdigit() else float('inf')  # Convert to integer
	return float('inf')  # Default if no number found