import qai_hub as hub

dataset_path = "datasets\dataset_1.h5"

# Submit inference job
inference_job = hub.submit_inference_job(
	model = hub.get_model("mn06pd49n"), # Using previously uploaded model
	device = hub.Device("Samsung Galaxy S23"),
	inputs = dataset_path,  # Pass the processed image tensor
)

# Ensure the job is valid
assert isinstance(inference_job, hub.InferenceJob)

output_data = inference_job.download_output_data(f"./h5_data/imagenet_set_1_data/wideresnet50_set_1")