import os
import shutil
import re

# Source folder containing files
source_folder = "./ILSVRC2012_img_val"  # Change this to your actual folder path

# Destination base folder
destination_base = "./split_imagenet_folders"

# Number of files per folder
files_per_folder = 1000

# Get list of all image files
files = os.listdir(source_folder)

# Extract the numeric part from filenames (last 8 digits)
def extract_number(filename):
    match = re.search(r'(\d{8})', filename)  # Look for exactly 8 digits
    return int(match.group()) if match else float('inf')  # Convert to integer

# Sort files based on extracted numbers
files = sorted(files, key=extract_number)

# Create and distribute files into folders
for index, file in enumerate(files):
    folder_index = index // files_per_folder  # Determines which folder to put the file in (0-based)

    folder_path = os.path.join(destination_base, f"imagenet_set_{folder_index + 1}")  # 1-based folder naming
    os.makedirs(folder_path, exist_ok=True)  # Create folder if it doesn't exist

    src_path = os.path.join(source_folder, file)
    dest_path = os.path.join(folder_path, file)
    
    shutil.move(src_path, dest_path)  # Move file

print(f"📂 Successfully split {len(files)} files into folders! 📂")