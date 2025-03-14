import qai_hub as hub

# Path to your .h5 file
dataset = "datasets\imagenet1k_set_0.h5"  # Replace with your actual file path

hub_dataset = hub.upload_dataset(dataset)

# Submit inference job
inference_job = hub.submit_inference_job(
	model = hub.get_model("mn06pd49n"), # Using previously uploaded model
	device = hub.Device("Samsung Galaxy S23"),
	inputs = hub_dataset,
)

inference_job.download_output_data("batch_dataset_0")