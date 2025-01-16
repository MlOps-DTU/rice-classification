import matplotlib.pyplot as plt
import torch
from data import get_rice_pictures
from model import RiceClassificationModel
import hydra

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path="../../configs", config_name="train.yaml", version_base=None)
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

    # Initialize the model and move it to the specified device
    model = RiceClassificationModel(num_classes=cfg.parameters.num_classes).to(DEVICE)
    
    # Load the training dataset
    train_set, _ = get_rice_pictures()

    # Create a DataLoader for batching and shuffling the training data
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=cfg.parameters.batch_size, shuffle=True)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # Dictionary to store training statistics
    statistics = {"train_loss": [], "train_accuracy": []}
    
    # Training loop for the specified number of epochs
    for epoch in range(cfg.parameters.epoch):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        # Iterate over batches of training data
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)  # Move data to the specified device
            optimizer.zero_grad()  # Zero the gradients
            y_pred = model(img)  # Forward pass
            loss = loss_fn(y_pred, target)  # Compute the loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update the model parameters

            epoch_loss += loss.item()  # Accumulate the loss
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()  # Compute accuracy
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

    print("Training complete")

    # Save the trained model to the models folder
    torch.save(model.state_dict(), "../../models/rice_model.pth")

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

    fig.savefig("../../reports/figures/training_curve.png")

if __name__ == "__main__":
    main()
