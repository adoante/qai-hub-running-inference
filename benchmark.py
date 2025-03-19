import threading

from one_script_to_rule_them_all import inference_dataset, process_results, calculate_accuracy, construct_datasets, InputSpec
from os import listdir

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

# Benchmark wideresnet50_quantized

model_id = "mnzrv83om"
device_name = "Samsung Galaxy S24 (Family)"
model_name = "wideresnet50_quantized"
library_name = "tflite"
results_dir = "benchmarked_model_datasets/wideresnet50_quantized_tflite_s24"

# for i in range(50):
# 	image_paths = [f"./imagenet_50k/imagenet_set_{i + 1}/{img}" for img in listdir(f"./imagenet_50k/imagenet_set_{i + 1}")]
# 	construct_datasets(image_paths, image_dataset_dir, i + 1, InputSpec.QUANTIZED)

image_dataset_dir = "image_datasets_quantized"
dataset_paths = [
		f"./{image_dataset_dir}/" + image_dataset 
		for image_dataset in listdir(f"./{image_dataset_dir}")
	]

dataset_paths.sort(key=extract_number)

# Split into two lists with alternating elements
list1 = dataset_paths[::2] # odd
list2 = dataset_paths[1::2] # even

thread1 = threading.Thread(
    target=inference_dataset,
    args=(list1, model_id, device_name, model_name, results_dir)
)

thread2 = threading.Thread(
    target=inference_dataset,
    args=(list2, model_id, device_name, model_name, results_dir)
)

thread1.start()
thread2.start()

thread1.join()
thread2.join()

result_paths = [f"./{results_dir}/" + result for result in listdir(f"./{results_dir}")]
result_paths.sort(key=extract_number)
process_results(result_paths, "mappings/class_index.json", "mappings/synset.json")

results_json = "results.json"
ground_truth_json = "./mappings/val_ground_truth.json"

calculate_accuracy(results_json, ground_truth_json, "Samsung Galaxy S24", model_name, library_name)