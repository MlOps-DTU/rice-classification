import torch
from torch import nn


class RiceClassificationModel(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for rice image classification.

    This model consists of three convolutional layers followed by max pooling layers,
    a dropout layer for regularization, and two fully connected layers for classification.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 3 input channels and 32 output channels.
        conv2 (nn.Conv2d): Second convolutional layer with 32 input channels and 64 output channels.
        conv3 (nn.Conv2d): Third convolutional layer with 64 input channels and 128 output channels.
        pool (nn.MaxPool2d): Max pooling layer with a kernel size of 2 and stride of 2.
        dropout (nn.Dropout): Dropout layer with a dropout probability of 0.5.
        fc1 (nn.Linear): First fully connected layer with input size 128*6*6 and output size 256.
        fc2 (nn.Linear): Second fully connected layer with input size 256 and output size num_classes.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Defines the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor representing a batch of images.
            Returns:
                torch.Tensor: Output tensor representing the class scores for each image in the batch.
    """
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    num_classes = 5 
    model = RiceClassificationModel(num_classes=num_classes)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test with dummy input
    dummy_input = torch.randn(1, 1, 50, 50)  # Batch size = 1, Channels = 1, Image size = 50x50
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
