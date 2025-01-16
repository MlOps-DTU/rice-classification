import torch
from rice_classification.data import get_rice_pictures
from rice_classification.model import RiceClassificationModel
import hydra
from loguru import logger
import os

# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="evaluate.yaml", version_base=None)
def main(cfg) -> None:
    """
    Main function to evaluate the rice classification model.

    Args:
        cfg: Configuration object containing parameters for the evaluation process.

    Returns:
        None

    The function performs the following steps:
    1. Initializes the rice classification model and moves it to the appropriate device.
    2. Loads the model weights from a pre-trained model.
    3. Loads the test dataset and creates a DataLoader for it.
    4. Sets the model to evaluation mode.
    5. Evaluates the model on the test dataset and calculates the accuracy.
    6. Prints the test accuracy.
    """

    
    # Specify the model and move it to the appropriate device
    logger.info("A model is initialized for evaluation.")
    model = RiceClassificationModel(num_classes=cfg.parameters.num_classes).to(DEVICE)

    # Load the model weights from the already trained model
    logger.info("The trained model is loaded for evaluation.")
    model.load_state_dict(torch.load(cfg.parameters.model_path, weights_only = True))

    # Load the test set
    _, test_set = get_rice_pictures()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.parameters.batch_size)

    # Set the model to evaluation mode
    model.eval()

    # Evaluate the model
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    # Calculate the accuracy
    accuracy = correct / total
    logger.info(f"The test accuracy is {accuracy:.4f}")
    print(f"Test accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    main()
