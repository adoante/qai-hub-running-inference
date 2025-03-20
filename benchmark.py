import threading

from one_script_to_rule_them_all import inference_dataset, process_results, calculate_accuracy, construct_datasets, InputSpec, extract_number, FileType
from os import listdir

# Benchmark wideresnet50_quantized

model_id = "mqy66g49m"
device_name = "Samsung Galaxy S24 (Family)"
model_name = "wideresnet50_quantized"
library_name = "onnx"
results_dir = "benchmarked_model_datasets/wideresnet50_quantized_onnx_s24"

### Construct image datasets

image_dataset_dir = "image_datasets/image_datasets_onnx"

# for i in range(50):
# 	image_paths = [f"./imagenet_50k/imagenet_set_{i + 1}/{img}" for img in listdir(f"./imagenet_50k/imagenet_set_{i + 1}")]
# 	construct_datasets(image_paths, image_dataset_dir, i + 1, InputSpec.NORMAL, FileType.ONNX)

### Inference Image datasets

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

### Process results from inference

result_paths = [f"./{results_dir}/" + result for result in listdir(f"./{results_dir}")]
result_paths.sort(key=extract_number)
process_results(result_paths, "mappings/class_index.json", "mappings/synset.json")

### Calculate accuracy based on processed results

results_json = "results.json"
ground_truth_json = "./mappings/val_ground_truth.json"

calculate_accuracy(results_json, ground_truth_json, "Samsung Galaxy S24", model_name, library_name)