import torch
import typer
from torchvision import datasets, transforms

def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


"../../data/raw"
"../../data/processed"

def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    datasets = []
    folders = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize((150, 150)),  # Resize all images to a fixed size
        normalize(),                    # Normalize the values of the images
        transforms.ToTensor(),          # Convert images to PyTorch tensors
    ])

    for folder in folders:
        data_dir = f"{raw_dir}/{folder}
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        datasets.append(dataset)
        
    train_images, train_target = [], []
    test_images, test_target = [], []

    for label, dataset in enumerate(datasets):
        train_images.append(dataset.data[:12500])
        test_images.append(dataset.data[12500:])
        train_target.append(torch.tensor(label)* 2500)
        test_target.append(torch.tensor(label)* 2500)

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_target, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_target, f"{processed_dir}/test_target.pt")

def get_rice_pictures() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for the rice."""
    train_images = torch.load("../../data/processed/train_images.pt")
    train_target = torch.load("../../data/processed/train_target.pt")
    test_images = torch.load("../../data/processed/test_images.pt")
    test_target = torch.load("../../data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set

if __name__ == "__main__":
    typer.run(preprocess_data)
