import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Register convolutional feature extractors and dense classifier layers.
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        flattened_features = 16 * 5 * 5
        self.fc1 = nn.Linear(flattened_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        # Extract convolutional features: (batch, 1, 32, 32) -> (batch, 16, 5, 5).
        conv1 = self.conv1(x)  # (batch, 1, 32, 32) -> (batch, 6, 28, 28)
        x = torch.tanh(conv1)  # (batch, 6, 28, 28)
        x = F.avg_pool2d(x, kernel_size=2)  # (batch, 6, 28, 28) -> (batch, 6, 14, 14)
        conv2 = self.conv2(x)  # (batch, 6, 14, 14) -> (batch, 16, 10, 10)
        x = torch.tanh(conv2)  # (batch, 16, 10, 10)
        x = F.avg_pool2d(x, kernel_size=2)  # (batch, 16, 10, 10) -> (batch, 16, 5, 5)

        # Flatten feature maps for dense layers: (batch, 16, 5, 5) -> (batch, 400).
        x = torch.flatten(x, start_dim=1)  # (batch, 16, 5, 5) -> (batch, 400)

        # Classify flattened features: (batch, 400) -> (batch, 10).
        fc1 = self.fc1(x)  # (batch, 400) -> (batch, 120)
        x = torch.tanh(fc1)  # (batch, 120)
        fc2 = self.fc2(x)  # (batch, 120) -> (batch, 84)
        x = torch.tanh(fc2)  # (batch, 84)
        logits = self.output(x)  # (batch, 84) -> (batch, 10)
        return logits
