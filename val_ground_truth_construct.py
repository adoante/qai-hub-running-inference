import json

# Define the input file name
input_filename = "./val_ground_truth.txt"

# Initialize an empty dictionary
data_dict = {}

# Read the file and store line numbers as keys
with open(input_filename, "r") as file:
    for line_number, line in enumerate(file, start=1):  # Line numbers start from 1
        data_dict[line_number] = line.strip()  # Remove newline characters

# {
# 	"img_id": "synset_id",
#    etc.
# }
with open("val_ground_truth.json", "w") as file:
    json.dump(data_dict, file)