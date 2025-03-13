import h5py
import numpy as np

output_file = "wideresnet50.h5"  # Replace with your file path

with h5py.File(output_file, "r") as f:
    # Navigate to the dataset
    dataset_path = "data/0/batch_0"
    
    if dataset_path in f:
        data = f[dataset_path][()]  # Read the dataset
        print(f"Data shape: {data.shape}, dtype: {data.dtype}")
    else:
        print(f"Dataset '{dataset_path}' not found in the file.")

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

probabilities = softmax(data)  # Apply softmax

# Load ImageNet class labels (downloaded JSON file)
imagenet_class_index_url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"

# Download and load class labels into a dictionary
import urllib.request
import json

with urllib.request.urlopen(imagenet_class_index_url) as url:
    class_idx = json.loads(url.read().decode())

# Get predicted class index (from previous step)
predicted_class = np.argmax(probabilities, axis=1)[0]

# Map the class index to class label
predicted_label = class_idx[str(predicted_class)][1]

print(f"Predicted class index: {predicted_class}")
print(f"Predicted class label: {predicted_label}")

# Print top five predictions for the on-device model
print("Top-5 classes")
top5_classes = np.argsort(probabilities[0], axis=0)[-5:]
for class_index in top5_classes:
    print(f"{class_index} {class_idx[str(class_index)][1]} {probabilities[0][class_index]:>6.1%}")