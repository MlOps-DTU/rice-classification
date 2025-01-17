from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import wandb
import torchvision.transforms as transforms
from loguru import logger
import os
from rice_classification.model import RiceClassificationModel
from dotenv import load_dotenv

# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")
load_dotenv()
wanb_api_key = os.getenv("WANDB_API_KEY")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CATEGORIES = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, transform

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
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Set the model to evaluation mode
    model.eval()

    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((50, 50)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    yield

    print("Cleaning up")
    del model

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"Hello and welcome to the rice classification app!"}

@app.post("/predict/")
async def predict(data: UploadFile = File(...)):
    """Make a prediction for an uploaded image."""
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    image = transform(i_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    category = CATEGORIES[predicted.item()]
    return {"Prediction of rice category": category}