import h5py

def print_batch_paths(group, prefix=""):
    """Recursively print paths of all batches (datasets) in the group."""
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            # If the item is a group, recurse into it
            print_batch_paths(item, prefix + key + "/")
        else:
            # If the item is a dataset, print the path
            print(f"{prefix}{key}")

# Open the HDF5 file
file_path = "h5_data\wideresnet50_set_1.h5"  # Replace with your file path
with h5py.File(file_path, 'r') as h5_file:
    # Print paths of all batches (datasets)
    print_batch_paths(h5_file)