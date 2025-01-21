from google.cloud import storage
import os
import torch
import numpy as np

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


BUCKET_NAME = "emils_mlops_data_bucket"
source_blob_name = "path/in/bucket/your_file.pt"
destination_dir = "data/processed"

def extract_features_batch(images):
    """Extract basic image features from a batch of images."""
    avg_brightness = np.mean(images, axis=(1, 2))  # Mean along the height and width axes
    contrast = np.std(images, axis=(1, 2))  # Standard deviation along the height and width axes
    sharpness = np.mean(np.abs(np.gradient(images, axis=(1, 2))), axis=(1, 2))  # Approximate sharpness
    
    return avg_brightness, contrast, sharpness

def download_from_gcs_if_not_exists(bucket_name, source_blob_name="train.pt", destination_dir = "data/processed"):
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Build the destination file path
    destination_file_name = os.path.join(destination_dir, os.path.basename(source_blob_name))

    if os.path.exists(destination_file_name):
        print(f"File already exists locally at {destination_file_name}. Skipping download.")
    else:
        # Initialize the GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        # Download the file
        blob.download_to_filename(destination_file_name)
        print(f"File {source_blob_name} downloaded to {destination_file_name}.")

    return destination_file_name


train_data = torch.load(download_from_gcs_if_not_exists(BUCKET_NAME), weights_only=False)
# Initialize lists to hold the features and labels
features = []
labels = []

# Create a DataLoader to iterate through the dataset
data_loader = DataLoader(train_data, batch_size=32, shuffle=False)


print(labels)