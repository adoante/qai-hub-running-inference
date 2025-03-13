import numpy as np
from PIL import Image
import qai_hub as hub

# Load and preprocess the image
image_path = "./images/lizzy2.jpg"  # Replace with the path to your image
image = Image.open(image_path)  # Open the image using PIL

# Resize the image to 224x224 (expected by WideResNet50)
image = image.resize((224, 224))

# Convert the image to numpy array and normalize if required (e.g., scaling to [0, 1] range)
image = np.array(image).astype(np.float32) / 255.0

# The model expects a batch dimension, so we add one (shape becomes (1, 224, 224, 3))
image = np.expand_dims(image, axis=0)

# Submit inference job
inference_job = hub.submit_inference_job(
    model="wideresnet50.tflite",
    device=hub.Device("Samsung Galaxy S23"),  # Ensure the device matches available ones
    inputs=dict(image_tensor=[image]),  # Pass the processed image tensor
)

# Ensure the job is valid
assert isinstance(inference_job, hub.InferenceJob)

# Download output data
output_data = inference_job.download_output_data('wideresnet50')