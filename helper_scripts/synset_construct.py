import scipy.io
import json

# Load the .mat file
mat_file_path = "synset.mat"
mat_data = scipy.io.loadmat(mat_file_path)

# Extract the 'synsets' field
synsets = mat_data['synsets']

# Loop through the synsets and print relevant information
synset_dict = dict()

for synset in synsets:
    ILSVRC2012_ID = str(synset["ILSVRC2012_ID"][0][0])  # Extract the ID
    ILSVRC2012_ID = ILSVRC2012_ID.replace("[","")
    ILSVRC2012_ID = ILSVRC2012_ID.replace("]","")
   
    WNID = str(synset['WNID'][0])  # Extract the WordNet ID
    WNID = WNID.replace("[","")
    WNID = WNID.replace("]","")
    WNID = WNID.replace("'","")

    synset_dict[WNID] = ILSVRC2012_ID 

# {
# 	"WNID": "synset_id",
#     etc,
# }
with open("synset.json", "w") as json_file:
	json.dump(synset_dict, json_file)