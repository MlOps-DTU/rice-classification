import matplotlib.pyplot as plt
import torch
from rice_classification.data import get_rice_pictures
from rice_classification.model import RiceClassificationModel
import hydra
from loguru import logger
from dotenv import load_dotenv
import os
import wandb
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score


# Load environment variables
load_dotenv()
wanb_api_key = os.getenv("WANDB_API_KEY")

# Add a logger to the script that logs messages to a file
logger.add("my_log.log", level="DEBUG", rotation="100 MB")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path=f"{os.getcwd()}/configs", config_name="train.yaml", version_base=None)
def main(cfg) -> None:
    """
    Train the rice classification model.

    Args:
        cfg: Configuration object containing parameters for training, such as 
             number of classes, batch size, number of epochs, and optimizer settings.

    Returns:
        None

    The function performs the following steps:
    1. Initializes the rice classification model and moves it to the specified device.
    2. Loads the training dataset and creates a DataLoader for batching and shuffling.
    3. Defines the loss function and optimizer.
    4. Trains the model for a specified number of epochs, recording loss and accuracy.
    5. Prints training progress and statistics.
    6. Saves the trained model to the models folder.
    7. Plots and saves the training loss and accuracy curves.
    """
    print("Training the rice classification model")
    logger.info("Training of the rice classification model has started")

    # Initialize the model and move it to the specified device
    model = RiceClassificationModel(num_classes=cfg.parameters.num_classes).to(DEVICE)
    
    # Load the training dataset
    train_set, _ = get_rice_pictures()

    # Create a DataLoader for batching and shuffling the training data
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.parameters.batch_size, shuffle=True)
    logger.info("The data was loaded for training")

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    logger.info("The loss function and optimizer were defined")

    # Dictionary to store training statistics
    statistics = {"train_loss": [], "train_accuracy": []}

    run = wandb.init(
        project="rice-classification", job_type = "train",
        config={"lr": cfg.optimizer.lr, "batch_size": cfg.parameters.batch_size, "epochs": cfg.parameters.epoch},
    )
    
    # Training loop for the specified number of epochs
    for epoch in range(cfg.parameters.epoch):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        preds, targets = [], []
        # Iterate over batches of training data
        for i, (img, target) in enumerate(train_dataloader):
            # Move data to the specified device
            img, target = img.to(DEVICE), target.to(DEVICE) 

            # Zero the gradients
            optimizer.zero_grad()  

            # Forward pass
            y_pred = model(img)  

            # Compute the loss
            loss = loss_fn(y_pred, target)  

            # Backward pass
            loss.backward()

            # Update the model parameters
            optimizer.step()
            
            # Accumulate the loss
            epoch_loss += loss.item()  

            # Compute accuracy
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()  

            # Log the loss and accuracy to Weights and Biases
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            # Save results
            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())

            epoch_accuracy += accuracy

            # Print training progress every 100 iterations
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        # Compute average loss and accuracy for the epoch
        epoch_loss /= len(train_dataloader)
        epoch_accuracy /= len(train_dataloader)
        statistics["train_loss"].append(epoch_loss)
        statistics["train_accuracy"].append(epoch_accuracy)

        # Print epoch statistics
        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")


    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)
    
    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    print("Training complete")
    logger.info("The training finished successfully")

    # Save the trained model to the models folder
    torch.save(model.state_dict(), "models/rice_model.pth")
    artifact = wandb.Artifact(
        name="rice_classification_model",
        type="model",
        description="A model trained to classify rice images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("models/rice_model.pth")
    run.log_artifact(artifact)

    logger.info("The trained model was saved to the models folder")

    # Plot and save the training loss and accuracy curves
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"], label="Train Loss")
    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(statistics["train_accuracy"], label="Train Accuracy")
    axs[1].set_title("Train Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    # Save the figure
    fig.savefig("reports/figures/training_curve.png")

    # Log the figure to Weights and Biases
    wandb.log({"training_curve": wandb.Image(fig)})

if __name__ == "__main__":
    main()
