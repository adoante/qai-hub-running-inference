import h5py

h5_data = "h5_data\imagenet_set_1_data\wideresnet50_set_1.h5"

def list_datasets(file_path):
    dataset_paths = []
    def recursively_collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_paths.append(name)

    with h5py.File(file_path, 'r') as h5_file:
        h5_file.visititems(recursively_collect)
    
    return dataset_paths

# Example usage
datasets = list_datasets(h5_data)
print(datasets)