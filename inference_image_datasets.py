import qai_hub as hub
from os import listdir
import threading
import re

def inference_dataset(dataset, model_id, device_name, model_name):
	print(f"😩 Inference: {dataset} 😩")
	# Submit inference job
	inference_job = hub.submit_inference_job(
		model = hub.get_model(model_id), # Using previously uploaded model
		device = hub.Device(device_name),
		inputs = dataset,  # Pass the processed image tensor
	)

	# Ensure the job is valid
	assert isinstance(inference_job, hub.InferenceJob)

	match = re.search(r'_(\d+)\.h5', dataset)
	number = int(match.group(1))
	inference_job.download_output_data(f"./inference_image_datasets/{model_name}_{number}")

# Sorting the filenames based on the numerical part after '_'
def extract_number(filename):
	parts = filename.split('_')  # Split by '_'

	if len(parts) > 1:  # Ensure there is a number part
		num_part = ''.join(filter(str.isdigit, parts[-1]))  # Extract digits
		return int(num_part) if num_part.isdigit() else float('inf')  # Convert to integer
	return float('inf')  # Default if no number found

dataset_paths = [f"./preprocess_image_datasets/{h5_file}" for h5_file in listdir(f"./preprocess_image_datasets")]
dataset_paths.sort(key=extract_number)  # Sort using the extracted numbers

# Split into two lists with alternating elements
list1 = dataset_paths[::2] # odd
list2 = dataset_paths[1::2] # even

# Create and start two threads, each processing one list
thread1 = threading.Thread(target=lambda: [inference_dataset(file, "mm5jvkj6n", "Samsung Galaxy S24 (Family)", "vit-snapdragon_8_elite") for file in list1])
thread2 = threading.Thread(target=lambda: [inference_dataset(file, "mm5jvkj6n", "Samsung Galaxy S24 (Family)", "vit-snapdragon_8_elite") for file in list2])

thread1.start()
thread2.start()

# Wait for both threads to finish
thread1.join()
thread2.join()