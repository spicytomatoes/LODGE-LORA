import argparse
import os
import shutil
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="finedance",
    help="dataset name",
)
parser.add_argument(
    "--split_perc",
    type=float,
    default=0.8,
    help="percentage of data to be used for training",
)

args = parser.parse_args()

# check if the dataset directory exists
dataset_dir = f"data/{args.dataset}/"
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

# check if the dataset directory has expected subdirectories
subdirs = os.listdir(f"data/{args.dataset}/")
expected_subdirs = ["label_json", "mofea319", "motion", "music_npy", "music_npynew", "music_wav"]

assert set(expected_subdirs).issubset(set(subdirs)), f"Expected subdirectories {expected_subdirs} not found in {dataset_dir}. Found: {subdirs}"

# create train and test directories
train_dir = f"data/{args.dataset}_train/"
test_dir = f"data/{args.dataset}_test/"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

num_files_last = None

for dir in expected_subdirs:
    dir_path = os.path.join(dataset_dir, dir)
    
    files = glob(os.path.join(dir_path, "*.*"))
    files = sorted(files)
    
    if num_files_last is None:
        num_files_last = len(files)
        
    if len(files) != num_files_last:
        raise ValueError(f"Number of files in {dir} does not match the number of files in the first directory, {num_files_last}. Found: {len(files)}")
    
    # Split the files into training and testing sets
    num_train = int(len(files) * args.split_perc)
    train_files = files[:num_train]
    test_files = files[num_train:]
    
    # Copy the files to the respective directories
    os.makedirs(os.path.join(train_dir, dir), exist_ok=True)
    os.makedirs(os.path.join(test_dir, dir), exist_ok=True)
    
    for file in train_files:
        shutil.copy(file, os.path.join(train_dir, dir, os.path.basename(file)))
    for file in test_files:
        shutil.copy(file, os.path.join(test_dir, dir, os.path.basename(file)))
        
print(f"Data split completed. {len(train_files)} files for training and {len(test_files)} files for testing.")
