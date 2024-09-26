import os
import cv2
import numpy as np

# Preprocess images for FER-2013 (48x48 grayscale)
def preprocess_fer_images(image_path, target_size=(48, 48)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
    image_resized = cv2.resize(image, target_size)        # Resize to 48x48
    image_normalized = image_resized / 255.0              # Normalize pixel values
    return image_normalized

# Preprocess images for UTKFace (64x64 color images)
def preprocess_utkface_images(image_path, target_size=(64, 64)):
    image = cv2.imread(image_path)                        # Load image as color
    image_resized = cv2.resize(image, target_size)        # Resize to 64x64
    image_normalized = image_resized / 255.0              # Normalize pixel values
    return image_normalized

# Load dataset and preprocess all images (FER example)
def load_fer_data(directory, target_size=(48, 48)):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory (ignores .DS_Store)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                if image_name.endswith('.png') or image_name.endswith('.jpg'):  # Filter valid image files
                    image_resized = preprocess_fer_images(image_path, target_size)
                    images.append(image_resized)
                    labels.append(label)  # Assuming labels are folder names
    return np.array(images), np.array(labels)

# Load dataset and preprocess all images (UTKFace example)
def load_utkface_data(directory, target_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory (ignores .DS_Store)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                if image_name.endswith('.png') or image_name.endswith('.jpg'):  # Filter valid image files
                    image_resized = preprocess_utkface_images(image_path, target_size)
                    images.append(image_resized)
                    labels.append(label)  # Assuming labels are folder names or age ranges
    return np.array(images), np.array(labels)
