import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import hydra
import sys
from loguru import logger

# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="data.yaml", version_base=None)
def main(cfg) -> None:
    """
    Process raw data and save it to the processed directory.
    Args:
        cfg: Configuration object containing parameters for data processing.
    The function performs the following steps:
    1. Defines transformations for the images including resizing, converting to tensor, and normalizing.
    2. Loads the dataset from the specified raw data directory.
    3. Retrieves the labels from the dataset.
    4. Splits the dataset into training and testing sets while ensuring class balance.
    5. Creates subsets for training and testing data.
    6. Saves the processed training and testing data to the specified processed data directory.
    """
    
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((cfg.parameters.height, cfg.parameters.width)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5]),                        
    ])

    # Load the dataset
    logger.info("The dataset is loaded from the raw data directory.")
    data_dir = f"{cfg.parameters.raw_dir}"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Get the labels
    targets = dataset.targets

    # Split the dataset into training and testing sets
    logger.info("The dataset is split into training and testing sets.")
    train_indices, test_indices = train_test_split(
    range(len(dataset)),
    stratify=targets,  # Ensures class balance
    test_size=cfg.parameters.test_size,     # 20% for testing
    )
    
    # Create subsets directly
    train = Subset(dataset, train_indices)
    test = Subset(dataset, test_indices)

    processed_data_dir = f"{cfg.parameters.processed_dir}"

    logger.info("The dataset is saved to the processed data directory.")
    torch.save(train, f"{processed_data_dir}/train.pt")
    torch.save(test, f"{processed_data_dir}/test.pt")

def get_rice_pictures() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads and returns the training and testing datasets for rice classification.
    The function loads the datasets from preprocessed files located at 
    "../../data/processed/train.pt" and "../../data/processed/test.pt".
    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]: 
            A tuple containing the training dataset and the testing dataset.
    """
    logger.info("The datasets are loaded from the processed data directory.")
    train_set = torch.load("data/processed/train.pt", weights_only=False)
    test_set = torch.load("data/processed/test.pt", weights_only=False)
    
    return train_set, test_set

if __name__ == "__main__":
    main()