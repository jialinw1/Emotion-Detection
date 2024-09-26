import os
import shutil
import random

# Define paths
source_dir = 'utkface-new/UTKFace'  # The directory containing all images
train_dir = 'utkface-new/train'  # Where you want to save the training data
test_dir = 'utkface-new/test'  # Where you want to save the testing data

# Create directories for age ranges in train and test folders
age_ranges = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
for age_range in age_ranges:
    os.makedirs(os.path.join(train_dir, age_range), exist_ok=True)
    os.makedirs(os.path.join(test_dir, age_range), exist_ok=True)


# Define a function to get the age range based on age
def get_age_range(age):
    age = int(age)
    if age <= 10:
        return '0-10'
    elif age <= 20:
        return '11-20'
    elif age <= 30:
        return '21-30'
    elif age <= 40:
        return '31-40'
    elif age <= 50:
        return '41-50'
    elif age <= 60:
        return '51-60'
    elif age <= 70:
        return '61-70'
    elif age <= 80:
        return '71-80'
    elif age <= 90:
        return '81-90'
    else:
        return '91-100'


# Get all files from the source directory
all_files = os.listdir(source_dir)

# Split ratio for train/test
split_ratio = 0.8
train_files = random.sample(all_files, int(len(all_files) * split_ratio))

# Split files into train and test sets
for file_name in all_files:
    # Skip non-image files
    if not file_name.endswith('.jpg'):
        continue

    # Extract the age from the file name (the first part of the file name)
    age = file_name.split('_')[0]

    # Get the age range based on the age
    age_range = get_age_range(age)

    # Define source file path
    src_file = os.path.join(source_dir, file_name)

    # Decide whether to put the file in the train or test set
    if file_name in train_files:
        dest_dir = os.path.join(train_dir, age_range)
    else:
        dest_dir = os.path.join(test_dir, age_range)

    # Move the file to the appropriate folder
    shutil.move(src_file, os.path.join(dest_dir, file_name))

print("Data splitting completed.")
