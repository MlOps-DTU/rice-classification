import os
import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import hydra
from loguru import logger
import subprocess
import zipfile


# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")


@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="data.yaml", version_base=None)
def main(cfg) -> None:
    """
    Process raw data and save it to the processed directory.
    Args:
        cfg: Configuration object containing parameters for data processing.
    The function performs the following steps:
    1. Checks if the raw data directory exists. If not, it downloads and unzips the raw data.
    2. Defines transformations for the images including resizing, converting to tensor, and normalizing.
    3. Loads the dataset from the specified raw data directory.
    4. Retrieves the labels from the dataset.
    5. Splits the dataset into training and testing sets while ensuring class balance.
    6. Creates subsets for training and testing data.
    7. Saves the processed training and testing data to the specified processed data directory.
    """

    # Define transformations for the images
    transform = transforms.Compose(
        [
            transforms.Resize((cfg.parameters.height, cfg.parameters.width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    if not os.path.isdir("data/raw"):
        logger.info("The raw data directory does not exist and will be downloaded first.")
        run_dvc_pull()
        # Unzip the raw data
        unzip_file(cfg.parameters.zip_file_path, cfg.parameters.raw_dir)

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
        test_size=cfg.parameters.test_size,  # 20% for testing
    )

    # Create subsets directly
    train = Subset(dataset, train_indices)
    test = Subset(dataset, test_indices)

    processed_data_dir = f"{cfg.parameters.processed_dir}"

    logger.info("The dataset is saved to the processed data directory.")
    torch.save(train, f"{processed_data_dir}/processed_data_train.pt")
    torch.save(test, f"{processed_data_dir}/processed_data_test.pt")


def get_rice_pictures() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Loads and returns the training and testing datasets for rice classification.
    The function loads the datasets from preprocessed files located at
    "../../data/processed/train.pt" and "../../data/processed/test.pt".
    Returns:
        tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            A tuple containing the training dataset and the testing dataset.
    """
    data_path_processed = "data/processed"
    os.makedirs(data_path_processed, exist_ok=True)
    # Build the destination file path
    destination_file_name_train = os.path.join(data_path_processed, "processed_data_train.pt")
    destination_file_name_test = os.path.join(data_path_processed, "processed_data_test.pt")

    if os.path.exists(destination_file_name_train) and os.path.exists(destination_file_name_test):
        logger.info("The datasets are loaded from the processed data directory.")  
    else:
        logger.info("The datasets are not found in the processed data directory and will be downloaded first")
        run_dvc_pull()
    
    train_set = torch.load("data/processed/processed_data_train.pt", weights_only=False)
    test_set = torch.load("data/processed/processed_data_test.pt", weights_only=False)
    return train_set, test_set

def run_dvc_pull():
    """
    Executes the `dvc pull` command to fetch data from the remote storage.

    This function runs the `dvc pull` command using the subprocess module to
    ensure that the data files tracked by DVC (Data Version Control) are
    up-to-date with the remote storage. If the command fails, an error is
    logged.

    Raises:
        subprocess.CalledProcessError: If the `dvc pull` command fails.
    """
    try:
        # Run the `dvc pull` command
        subprocess.run(["dvc", "pull", "-v"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC Pull Failed: {e}")

def unzip_file(zip_file_path="data/raw/data_raw_Arborio.zip", extract_to_path="data/raw_unzipped"):
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
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract all contents
            zip_ref.extractall(extract_to_path)
            logger.info("The raw zip has succesfull been unpacked.")
    except FileNotFoundError:
        logger.error(f"Error: The file {zip_file_path} does not exist.")
    except zipfile.BadZipFile:
        logger.error(f"Error: The file {zip_file_path} is not a valid zip file.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()