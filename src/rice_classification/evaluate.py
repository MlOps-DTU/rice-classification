import torch
from rice_classification.data import get_rice_pictures
from rice_classification.model import RiceClassificationModel
import hydra

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@hydra.main(config_path="../../configs", config_name="evaluate.yaml", version_base=None)
def main(cfg) -> None:
    model = RiceClassificationModel(num_classes=cfg.parameters.num_classes).to(DEVICE)
    model.load_state_dict(torch.load(cfg.parameters.model_path, weights_only = True))

    _, test_set = get_rice_pictures()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=cfg.parameters.batch_size)

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
    main()
