import h5py
import numpy as np
import json
from os import listdir

# Path to h5 files
output_paths = ["./h5_data/" + img for img in listdir("./h5_data")]

# Dictionary to store results
data = dict()
data_key = 1

# Load ImageNet class indexes
with open('class_index.json', 'r') as json_file:
    class_indexes = json.load(json_file)

# Load Synset JSON
with open("synset.json", "r") as file:
    synset_dict = json.load(file)

# Softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Prevents overflow
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Process each h5 file
for output_file in output_paths:
    with h5py.File(output_file, "r") as h5_file:
        # Assuming dataset is named as 'data/0/batch_i' where i is the batch index
        for batch_idx in range(1000):  # 1000 batches
            dataset_path = f"data/0/batch_{batch_idx}"

            if dataset_path in h5_file:
                logits = h5_file[dataset_path][()]  # Read logits for batch

                # Apply softmax
                probabilities = softmax(logits)

                # Get Top 5 Predictions
                top5_predictions = np.argsort(probabilities[0])[-5:][::-1]  # Descending order

                # Store top 5 synsets
                top5_results = []

                for class_index in top5_predictions:
                    wnid = class_indexes[str(class_index)][0]  # Get synset ID
                    synset = synset_dict.get(wnid, "Unknown")  # Avoid KeyError
                    top5_results.append(str(synset))

                # Add results to dictionary
                data[str(data_key)] = top5_results
                data_key += 1
            else:
                print(f"Skipping {dataset_path}, dataset path not found.")

# Save results as JSON
with open("val_results.json", "w") as json_file:
    json.dump(data, json_file, indent=4)

print("Results saved to val_results.json")
