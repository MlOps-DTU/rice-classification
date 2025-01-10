import torch
import typer
from data import get_rice_pictures
from model import RiceClassificationModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def evaluate(model_checkpoint: str, num_classes: int = 5) -> None:
    """Evaluate a trained model."""
    print("Evaluating the rice classification model")
    print(model_checkpoint)

    model = RiceClassificationModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = get_rice_pictures()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()

    # Evaluate the model
    correct, total = 0, 0
    with torch.no_grad():
        for img, target in test_dataloader:
            img, target = img.to(DEVICE), target.to(DEVICE)
            y_pred = model(img)
            correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    print(f"Test accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    typer.run(evaluate)
