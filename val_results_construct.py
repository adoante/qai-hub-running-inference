import h5py
import numpy as np
import json
from os import listdir

# Path to outputs
output_paths = [("./h5_data/" + img) for img in listdir("./h5_data")]

# Will hold every outputs data
data = dict()
data_key = 1

for output_file in output_paths:

	with h5py.File(output_file, "r") as h5_file:
		# Navigate to the dataset
		dataset_path = "data/0/batch_1"
		
		# Read the dataset
		logits  = h5_file[dataset_path][()]  

	# Softmax function
	def softmax(x):
		return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

	# Apply softmax to logits
	probabilities = softmax(logits)

	# Load ImageNet class indexes
	with open('class_index.json', 'r') as json_file:
		class_indexes = json.load(json_file)

	# 5 results
	top5_results = []

	# Top 5 Predictions
	top5_predictions = np.argsort(probabilities[0], axis=0)[-5:][::-1]

	# Load Synset JSON from the file
	with open("synset.json", "r") as file:
		synset_dict = json.load(file)

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