from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from PIL import Image
import wandb
import torchvision.transforms as transforms
from loguru import logger
import os
from src.rice_classification.model import RiceClassificationModel
from dotenv import load_dotenv
from datetime import datetime
import pytz
import numpy as np
from google.cloud import storage
import json

# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")
load_dotenv()
wanb_api_key = os.getenv("WANDB_API_KEY")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CATEGORIES = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

BUCKET_NAME = "prediction_meta_data"

def extract_features(images):
    """Extract basic image features from a set of images."""
    features = []
    for img in images:
        avg_brightness = np.mean(img)
        contrast = np.std(img)
        sharpness = np.mean(np.abs(np.gradient(img)))
        features.append([avg_brightness, contrast, sharpness])
    return np.array(features)

# Save prediction results to GCP
def save_prediction_to_gcp(avg_brightness:float, contrast:float, sharpness:float, label:int):
    """Save the prediction results to GCP bucket."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    copenhagen_tz = pytz.timezone('Europe/Copenhagen')
    time = datetime.now(copenhagen_tz)
    # Prepare prediction data
    data = {
        "avg_brightness": avg_brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "label": label,
        "timestamp": time.isoformat()
    }
    blob = bucket.blob(f"prediction_{time}.json")
    blob.upload_from_string(json.dumps(data))
    print("Prediction saved to GCP bucket.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    print("Loading model")

    # Initialize a new run
    run = wandb.init()

    # Use the specified artifact
    artifact = run.use_artifact('dtumlops25/rice-classification/rice_classification_model:v2', type='model')

    # Download the artifact
    artifact_dir = artifact.download()

    # Load the model from the artifact directory
    model_path = os.path.join(artifact_dir, 'rice_model.pth')
    
    # Initialize the model architecture
    model = RiceClassificationModel(num_classes=5).to(DEVICE)  # Adjust num_classes as needed
    
    # Load the state dict into the model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only = False))

    # Set the model to evaluation mode
    model.eval()

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Store model and transform in app.state for global access
    app.state.model = model
    app.state.transform = transform

    # Create a CSV file to store predictions
    with open("prediction_database.csv", "w") as file:
        file.write("time, image, prediction\n")

    yield

    # Cleanup
    print("Cleaning up")
    del app.state.model
    del app.state.transform

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"Hello and welcome to the rice classification app!"}

@app.post("/predict/")
async def predict(background_tasks: BackgroundTasks, data: UploadFile = File(...), ):
    """Make a prediction for an uploaded image."""
    # Access the model and transform from app.state
    model = app.state.model
    transform = app.state.transform

    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    image = transform(i_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    category = CATEGORIES[predicted.item()]

    image_features = extract_features(image.cpu().numpy())[0]

    background_tasks.add_task(save_prediction_to_gcp, float(image_features[0]), float(image_features[1]), float(image_features[2]), predicted.item())

    return {"Prediction of rice category": category, "Prediction int: ": predicted.item()}