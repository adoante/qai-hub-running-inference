import h5py
import numpy as np
import json

# Load ImageNet class indexes
with open('class_index.json', 'r') as json_file:
	class_indexes = json.load(json_file)

# Load Synset JSON from the file
with open("synset.json", "r") as file:
	synset_dict = json.load(file)

# HDF5 file path
h5_file_path = "h5_data\imagenet_set_1_data\wideresnet50_set_1.h5"

# Will hold every output data
data = dict()
data_key = 1

with h5py.File(h5_file_path, "r") as h5_file:
	# Find all dataset paths dynamically
	dataset_paths = []
	
	for i in range(1000):
		dataset_paths.append(f"data/0/batch_{i}")

	# Process each dataset found
	for dataset_path in dataset_paths:
		print(f"Reading dataset: {dataset_path}")
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

# Save results to JSON file
with open("val_results.json", "w") as json_file:
	json.dump(data, json_file)

print("Processing complete. Results saved to val_results.json.")
