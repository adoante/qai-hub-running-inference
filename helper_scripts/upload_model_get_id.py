import qai_hub as hub

model_file = "vit-snapdragon_8_elite.tflite"

model = hub.upload_model(model_file)
print(model.model_id)