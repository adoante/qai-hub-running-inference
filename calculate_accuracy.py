import json

# Load JSON files

with open("./mappings/val_ground_truth.json", "r") as file:
	ground_truth = json.load(file)
	
with open("val_results.json", "r") as file:
	val_results = json.load(file)

# Initialize counters for accuracy
correct_top1 = 0
correct_top5 = 0
total_images = len(val_results)

# Calculate top 1
for i in range(0, total_images):
	# Get top 1 and top 5 predictions
	top5_predictions = val_results[str(i + 1)]
	top1_prediction = top5_predictions[0]
	
	# Get ground truth
	gt = ground_truth[str(i + 1)]
	
	# Update accuracy counters
	if top1_prediction == gt:
		correct_top1 += 1

	if gt in top5_predictions:
		correct_top5 += 1

# Calculate top 1 and top 5 accuracy
top1_accuracy = round((correct_top1 / total_images) * 100, 2)
top5_accuracy = round((correct_top5 / total_images) * 100, 2)

accuracy_results = (top1_accuracy, top5_accuracy, "Samsung Galaxy S24", "squeezenet1_1", "tflite")

print(str(accuracy_results) + " 😀")