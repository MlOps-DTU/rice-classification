import torch
import torch.optim as optim
import torch.nn as nn
import pytest
from src.rice_classification.model import RiceClassificationModel 


def test_model_construction():
    """
    Test the RiceClassificationModel construction.
    This test checks if the model can be instantiated, has the correct architecture, and the expected number of parameters.
    """
    num_classes = 5  
    model = RiceClassificationModel(num_classes=num_classes)

    assert model is not None, "Model instantiation failed."

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params > 0, "Model has no parameters."

    assert isinstance(model.conv1, nn.Conv2d), "conv1 is not a Conv2d layer."
    assert isinstance(model.conv2, nn.Conv2d), "conv2 is not a Conv2d layer."
    assert isinstance(model.conv3, nn.Conv2d), "conv3 is not a Conv2d layer."
    assert isinstance(model.fc1, nn.Linear), "fc1 is not a Linear layer."
    assert isinstance(model.fc2, nn.Linear), "fc2 is not a Linear layer."
    assert isinstance(model.dropout, nn.Dropout), "dropout is not a Dropout layer."


def test_model_training():
    """
    Test if the RiceClassificationModel can train for one step with dummy data.
    This test checks if the model can perform a forward and backward pass without errors.
    """
    num_classes = 5
    model = RiceClassificationModel(num_classes=num_classes)

    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 50, 50)  
    dummy_labels = torch.randint(0, num_classes, (batch_size,))  

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    output = model(dummy_images)

    loss = criterion(output, dummy_labels)
    assert loss.item() > 0, f"Loss function returned invalid loss: {loss.item()}"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for param in model.parameters():
        assert param.grad is not None, "Gradients are None, model not learning."

    assert output.shape == (batch_size, num_classes), f"Expected output shape ({batch_size}, {num_classes}), but got {output.shape}."


if __name__ == "__main__":
    pytest.main()



"""
COVERAGE REPORT

>> coverage run -m pytest test_model.py      
======================================================= test session starts ========================================================
platform win32 -- Python 3.12.0, pytest-8.3.4, pluggy-1.5.0
rootdir: C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification
configfile: pyproject.toml
plugins: anyio-4.8.0, hydra-core-1.3.2
collected 2 items                                                                                                                   

test_model.py ..                                                                                                              [100%]

======================================================== 2 passed in 4.66s ========================================================= 


>> coverage report
Name                                                                                            Stmts   Miss  Cover
-------------------------------------------------------------------------------------------------------------------
C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/src/rice_classification/__init__.py       0      0   100%
C:/Users/USUARIO/Documents/DTU/MLOps2/rice-classification/src/rice_classification/model.py         32      7    78%
__init__.py                                                                                         4      0   100%
test_model.py                                                                                      36      1    97%
-------------------------------------------------------------------------------------------------------------------
TOTAL                                                                                              72      8    89%

"""
