import matplotlib.pyplot as plt
import torch
import typer
import os
from data import get_rice_pictures
from model import RiceClassificationModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 1, num_classes: int = 5) -> None:
    """Train the rice classification model."""
    print("Training the rice classification model")
    print(f"{lr=}, {batch_size=}, {epochs=}, {num_classes=}")

    model = RiceClassificationModel(num_classes=num_classes).to(DEVICE)
    train_set, _ = get_rice_pictures()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            epoch_accuracy += accuracy

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        epoch_loss /= len(train_dataloader)
        epoch_accuracy /= len(train_dataloader)
        statistics["train_loss"].append(epoch_loss)
        statistics["train_accuracy"].append(epoch_accuracy)

        print(f"Epoch {epoch}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.4f}")

    print("Training complete")

    # Save the model in the models folder
    torch.save(model.state_dict(), "../../models/rice_model.pth")

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
    typer.run(train)
