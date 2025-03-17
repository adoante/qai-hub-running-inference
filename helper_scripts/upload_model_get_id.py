import qai_hub as hub

model_file = "./shufflenet_v2.tflite"

model = hub.upload_model(model_file)
print(model.model_id)