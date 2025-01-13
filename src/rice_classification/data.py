import torch
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import hydra

@hydra.main(config_path="../../configs", config_name="data.yaml", version_base=None)
def main(cfg) -> None:
    """Process raw data and save it to processed directory."""
   
    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((cfg.parameters.height, cfg.parameters.width)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.5], std=[0.5]),                        
    ])

    # Load the dataset
    data_dir = f"{cfg.parameters.raw_dir}"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Get the labels
    targets = dataset.targets

    # Split the dataset into training and testing sets
    train_indices, test_indices = train_test_split(
    range(len(dataset)),
    stratify=targets,  # Ensures class balance
    test_size=cfg.parameters.test_size,     # 20% for testing
    )
    
    # Create subsets directly
    train = Subset(dataset, train_indices)
    test = Subset(dataset, test_indices)

    processed_data_dir = f"{cfg.parameters.processed_dir}"

    torch.save(train, f"{processed_data_dir}/train.pt")
    torch.save(test, f"{processed_data_dir}/test.pt")

def get_rice_pictures() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for the rice."""
    train_set = torch.load("../../data/processed/train.pt", weights_only=False)
    test_set = torch.load("../../data/processed/test.pt", weights_only=False)
    
    return train_set, test_set

if __name__ == "__main__":
    main()