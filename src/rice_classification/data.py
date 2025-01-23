import os
import subprocess
import zipfile

import hydra
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
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
        run_dvc_pull()
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

    logger.info("The dataset is saved to the processed data directory.")
    torch.save(train, f"{processed_data_dir}/train.pt")
    torch.save(test, f"{processed_data_dir}/test.pt")


def get_rice_pictures() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
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
        main()

    train_set: torch.utils.data.Dataset = torch.load("data/processed/train.pt", weights_only=False)
    test_set: torch.utils.data.Dataset = torch.load("data/processed/test.pt", weights_only=False)
    return train_set, test_set


def run_dvc_pull() -> None:
    """
    Executes the `dvc pull` command to fetch data from the remote storage.

    Parameters:
        None

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the `dvc pull` command fails.
    """
    try:
        # Run the `dvc pull` command
        subprocess.run(["dvc", "pull", "-v"], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC Pull Failed: {e}")


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


if __name__ == "__main__":
    main()
