import h5py
import numpy as np
import json
from os import listdir

# Load ImageNet class indexes
with open("mappings\class_index.json", "r") as file:
	class_indexes = json.load(file)

# Load Synset JSON from the file
with open("mappings\synset.json", "r") as file:
	synset_dict = json.load(file)

# Will hold every output data
data = dict()
data_key = 1

def val_results_construct(h5_file_path):
	global data
	global data_key

	print(f"Reading dataset: {h5_file_path}")

	with h5py.File(h5_file_path, "r") as h5_file:
		# Find all dataset paths dynamically
		dataset_paths = []
		
		for i in range(1000):
			dataset_paths.append(f"data/0/batch_{i}")

		# Process each dataset found
		for dataset_path in dataset_paths:
			
			logits = h5_file[dataset_path][()]  

			# Softmax function
			def softmax(x):
				return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

			# Apply softmax to logits
			probabilities = softmax(logits)

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

# Sorting the filenames based on the numerical part after '_'
def extract_number(filename):
	parts = filename.split('_')  # Split by '_'

	if len(parts) > 1:  # Ensure there is a number part
		num_part = ''.join(filter(str.isdigit, parts[-1]))  # Extract digits
		return int(num_part) if num_part.isdigit() else float('inf')  # Convert to integer
	return float('inf')  # Default if no number found

h5_data_files = listdir("./inference_image_datasets")
h5_data_files.sort(key=extract_number)  # Sort using the extracted numbers

for h5_file in h5_data_files:
	val_results_construct(f"./inference_image_datasets/{h5_file}")

# Save results to JSON file
with open("val_results.json", "w") as json_file:
	json.dump(data, json_file)
