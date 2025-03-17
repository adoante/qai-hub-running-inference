import qai_hub as hub
import re

dataset = "preprocess_image_datasets\dataset_5.h5"
model_id = "mq3j3dzln"
device_name = "Samsung Galaxy S24 (Family)"
model_name = "squeezenet1_1"

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