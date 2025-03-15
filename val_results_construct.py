import h5py
import numpy as np
import json
from os import listdir

# Load ImageNet class indexes
with open('class_index.json', 'r') as json_file:
	class_indexes = json.load(json_file)

# Load Synset JSON from the file
with open("synset.json", "r") as file:
	synset_dict = json.load(file)

# Path to outputs
h5_data_paths = []
origin_dir = listdir("./h5_data/imagenet_set_1_data/")

# Sorting the filenames based on the numerical part after '_'
def extract_number(filename):
    parts = filename.split('_')  # Split by '_'
    if len(parts) > 1:  # Ensure there is a number part
        num_part = ''.join(filter(str.isdigit, parts[-1]))  # Extract digits
        return int(num_part) if num_part.isdigit() else float('inf')  # Convert to integer
    return float('inf')  # Default if no number found

origin_dir.sort(key=extract_number)  # Sort using the extracted numbers

for image in origin_dir:
	h5_data_paths.append("./h5_data/imagenet_set_1_data/" + image)

# Will hold every outputs data
data = dict()
data_key = 1

for h5_data in h5_data_paths:
	print(f"Constructing: {h5_data} as JSON.")
	with h5py.File(h5_data, "r") as h5_file:
		# Navigate to the dataset
		dataset_path = "data/0/batch_0"
		
		# Read the dataset
		logits  = h5_file[dataset_path][()]  

	# Softmax function
	def softmax(x):
		return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

	# Apply softmax to logits
	probabilities = softmax(logits)

	# 5 results
	top5_results = []

	# Top 5 Predictions
	top5_predictions = np.argsort(probabilities[0], axis=0)[-5:][::-1]

	# Construct Top 5 results list
	for class_index in top5_predictions:
		# Find synset id
		wnid = class_indexes[str(class_index)][0]
		synset = synset_dict[wnid]

		top5_results.append(str(synset))

	# Add Top 5 results to outputs
	data[str(data_key)] = top5_results

	data_key += 1

# {
# 	"img_id": ["synset_id","synset_id","synset_id","synset_id","synset_id"],
#    etc.
# }

with open("val_results.json", "w") as json_file:
	json.dump(data, json_file)