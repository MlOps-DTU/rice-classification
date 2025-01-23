import torch

from rice_classification.data import get_rice_pictures

train_set, _ = get_rice_pictures()

data_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
# Create an iterator

for batch in data_loader:
    print(1)
