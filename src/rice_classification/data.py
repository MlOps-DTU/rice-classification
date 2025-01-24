import os
import random
import subprocess
import zipfile

import hydra
import numpy as np
import pandas as pd
import torch
from google.cloud import storage
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="data.yaml", version_base=None)
def main(cfg: dict) -> None:
    """
    Process raw data and save it to the processed directory.

    Args:
        cfg (dict): Configuration object containing parameters for data processing.

    Parameters:
        cfg.parameters.height (int): Height to resize the images.
        cfg.parameters.width (int): Width to resize the images.
        cfg.parameters.zip_file_path (str): Path to the zip file containing raw data.
        cfg.parameters.raw_dir (str): Directory to store raw data.
        cfg.parameters.test_size (float): Proportion of the dataset to include in the test split.
        cfg.parameters.processed_dir (str): Directory to store processed data.

    Returns:
        None
    """
    # Define transformations for the images
    transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((cfg.parameters.height, cfg.parameters.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    if not os.path.isdir("data/raw"):
        logger.info("The raw data directory does not exist and will be downloaded first.")
        run_gbucket()
        # Unzip the raw data
        unzip_file(cfg.parameters.zip_file_path, cfg.parameters.raw_dir)

    # Load the dataset
    logger.info("The dataset is loaded from the raw data directory.")
    data_dir: str = f"{cfg.parameters.raw_dir}"
    dataset: datasets.ImageFolder = datasets.ImageFolder(root=data_dir, transform=transform)

    # Get the labels
    targets: list[int] = dataset.targets

    # Split the dataset into training and testing sets
    logger.info("The dataset is split into training and testing sets.")
    train_indices: list[int]
    test_indices: list[int]
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        stratify=targets,  # Ensures class balance
        test_size=cfg.parameters.test_size,  # 20% for testing
    )

    # Create subsets directly
    train: Subset = Subset(dataset, train_indices)
    test: Subset = Subset(dataset, test_indices)

    processed_data_dir: str = f"{cfg.parameters.processed_dir}"
    os.makedirs(processed_data_dir, exist_ok=True)

    logger.info("The dataset is saved to the processed data directory.")
    torch.save(train, f"{processed_data_dir}/train.pt")
    torch.save(test, f"{processed_data_dir}/test.pt")


def get_rice_pictures(cfg: dict) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads and returns the training and testing datasets for rice classification.

    Parameters:
        None

    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            A tuple containing the training dataset and the testing dataset.
    """
    data_path_unzipped: str = "data/raw_unzipped"
    data_path_processed: str = "data/processed"
    if os.path.exists(data_path_unzipped) and os.path.exists(data_path_processed):
        logger.info("The datasets are loaded from the processed data directory.")
    else:
        logger.info("The datasets are not found in the processed data directory and will be downloaded first")
        main(cfg)

    train_set: torch.utils.data.Dataset = torch.load("data/processed/train.pt", weights_only=False)
    test_set: torch.utils.data.Dataset = torch.load("data/processed/test.pt", weights_only=False)
    return train_set, test_set


def run_gbucket() -> None:
    # Initialize a client
    client = storage.Client()

    # Define the bucket name and the object name
    bucket_name = "emils_mlops_data_bucket"
    folder_name = "data/raw/"
    destination_dir = "data/raw"

    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Get the bucket and the blob
    bucket = client.bucket(bucket_name)

    # List all the blobs in the specified folder
    blobs = bucket.list_blobs(prefix=folder_name)

    print("Downloading files from the 'raw' folder:")
    for blob in blobs:
        # Define the local file path
        local_file_path = os.path.join(destination_dir, os.path.basename(blob.name))

        # Download the blob to the local file
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")

    print("All files downloaded successfully.")

def unzip_file(zip_file_path: str = "data/raw/Arborio.zip", extract_to_path: str = "data/raw_unzipped") -> None:
    """
    Unzips a specified zip file to a given directory.

    Parameters:
        zip_file_path (str): The path to the zip file to be extracted. Default is "data/raw/Arborio.zip".
        extract_to_path (str): The directory where the contents of the zip file will be extracted. Default is "data/raw_unzipped".

    Returns:
        None
    """
    try:
        # Ensure the output directory exists
        os.makedirs(extract_to_path, exist_ok=True)

        # Open the zip file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            # Extract all contents
            zip_ref.extractall(extract_to_path)
            logger.info("The raw zip has succesfull been unpacked.")
    except FileNotFoundError:
        logger.error(f"Error: The file {zip_file_path} does not exist.")
    except zipfile.BadZipFile:
        logger.error(f"Error: The file {zip_file_path} is not a valid zip file.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def metrics_from_images(data_loader):
    """
    Processes a PyTorch DataLoader of images, extracts features in a batch, and converts them into a DataFrame.

    Args:
        data_loader (DataLoader): A DataLoader object containing the dataset of images and labels.

    Returns:
        pd.DataFrame: A DataFrame containing extracted image features and target labels.
    """
    # Prepare lists to store features and labels
    avg_brightness_list, contrast_list, sharpness_list = [], [], []
    labels_list = []

    # Iterate through the DataLoader
    for images, labels in data_loader:
        # Convert images to NumPy arrays if they are PyTorch tensors
        images_np = images.numpy() if torch.is_tensor(images) else images

        # Extract features for the entire batch at once
        avg_brightness, contrast, sharpness = extract_features_batch(images_np)

        # Append features and labels
        avg_brightness_list.extend(avg_brightness)
        contrast_list.extend(contrast)
        sharpness_list.extend(sharpness)
        labels_list.extend(labels.tolist())

    # Create a DataFrame
    dataframe = pd.DataFrame(
        {
            "avg_brightness": avg_brightness_list,
            "contrast": contrast_list,
            "sharpness": sharpness_list,
            "target": labels_list,
        }
    )

    return dataframe


def extract_features_batch(images):
    """
    Extract basic image features for an entire batch using vectorized operations.

    Args:
        images (np.array): A batch of images as a NumPy array of shape (batch_size, height, width, channels).

    Returns:
        tuple: Three NumPy arrays representing avg_brightness, contrast, and sharpness for the batch.
    """
    # Average brightness: Mean pixel value along axes (1, 2, 3) for height, width and channels
    avg_brightness = np.mean(images, axis=(1, 2, 3))

    # Contrast: Standard deviation along axes (1, 2, 3) for height, width and channels
    contrast = np.std(images, axis=(1, 2, 3))

    # Sharpness: Mean of absolute gradient along height, width and channels
    gradients = np.gradient(images, axis=(1, 2, 3))  # Gradient along , width and channels
    sharpness = np.mean(np.sqrt(gradients[0] ** 2 + gradients[1] ** 2), axis=(1, 2, 3))

    return avg_brightness, contrast, sharpness


def upload_to_bucket(source_file_name):
    """
    Uploads a file to a Google Cloud Storage bucket.

    Args:
        source_file_name (str): The path to the file to upload.
    """
    # Define the bucket name and the object name
    bucket_name = "emils_mlops_data_bucket"
    destination_blob_name = "data/processed/metric.csv"

    # Initialize a client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def process_and_save_metrics():
    """
    Creates a DataLoader from the dataset, processes the images, saves the metrics to a CSV file,
    and uploads the CSV file to a Google Cloud Storage bucket.
    """
    # Define the save path for the CSV file
    save_path = "data/processed/metric.csv"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load the dataset
    data = torch.load("data/processed/train.pt", weights_only=False)

    # Define a subset (e.g., first 500 samples or random samples)
    subset_size = 2000
    subset_indices = random.sample(range(len(data)), subset_size)
    subset_data = Subset(data, subset_indices)

    # Create a DataLoader
    data_loader = DataLoader(subset_data, batch_size=32, shuffle=False)

    # Extract metrics from images
    metrics_df = metrics_from_images(data_loader)

    # Save the DataFrame to a CSV file
    metrics_df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")

    # Upload the CSV file to the cloud bucket
    upload_to_bucket(save_path)


if __name__ == "__main__":
    main()
