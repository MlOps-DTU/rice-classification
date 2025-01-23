import json
import random
from contextlib import asynccontextmanager

import anyio
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
from loguru import logger
from torch.utils.data import DataLoader, Subset

from rice_classification.data import get_rice_pictures

# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")
load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CATEGORIES = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

BUCKET_NAME = "prediction_meta_data"

def download_data_from_gcp() -> pd.DataFrame:
    """Download data from GCP bucket and create a dataframe."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix="prediction_"))  # Convert to list

    # Create lists to store the data
    avg_brightness, contrast, sharpness, label, timestamp = [], [], [], [], []

    # Iterate over the blobs and collect their information
    for blob in blobs:
        data = json.loads(blob.download_as_string())
        avg_brightness.append(data["avg_brightness"])
        contrast.append(data["contrast"])
        sharpness.append(data["sharpness"])
        label.append(data["label"])

    # Create a DataFrame from the collected data
    dataframe = pd.DataFrame({
        "avg_brightness": avg_brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "target": label,
    })

    return dataframe

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
    dataframe = pd.DataFrame({
        "avg_brightness": avg_brightness_list,
        "contrast": contrast_list,
        "sharpness": sharpness_list,
        "target": labels_list
    })
    
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
    sharpness = np.mean(np.sqrt(gradients[0]**2 + gradients[1]**2), axis=(1, 2, 3))

    return avg_brightness, contrast, sharpness

def run_analysis(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Run the analysis and return the report."""
    data_drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
    data_drift_report.run(current_data=reference_data, reference_data=current_data)
    data_drift_report.save_html("monitoring.html")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("Loading Data")

    global train_metrics
    #  Get data from the data module
    training_data, _ = get_rice_pictures()

    # Define a subset (e.g., first 500 samples or random samples)
    subset_size = 2000
    subset_indices = random.sample(range(len(training_data)), subset_size) 
    subset_data = Subset(training_data, subset_indices)

    # Create a DataLoader for the subset
    data_loader = DataLoader(subset_data, batch_size=256, shuffle=False)
    train_metrics = metrics_from_images(data_loader)

    yield
    # Cleanup
    print("Cleaning up")
    del training_data, train_metrics

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"Hello and welcome to the rice classification monitoring app!"}

@app.get("/report", response_class=HTMLResponse)
async def get_report():
    """Generate and return the report."""

    prediction_data = download_data_from_gcp()
    run_analysis(train_metrics, prediction_data)

    async with await anyio.open_file("monitoring.html", mode="r", encoding="utf-8") as f:
        html_content = await f.read()

    return HTMLResponse(content=html_content, status_code=200)