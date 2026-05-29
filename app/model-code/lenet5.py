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
        conv1 = self.conv1(x)
        x = torch.tanh(conv1)
        x = F.avg_pool2d(x, kernel_size=2)
        conv2 = self.conv2(x)
        x = torch.tanh(conv2)
        x = F.avg_pool2d(x, kernel_size=2)

        # Flatten feature maps for dense layers: (batch, 16, 5, 5) -> (batch, 400).
        x = torch.flatten(x, start_dim=1)

        # Classify flattened features: (batch, 400) -> (batch, 10).
        fc1 = self.fc1(x)
        x = torch.tanh(fc1)
        fc2 = self.fc2(x)
        x = torch.tanh(fc2)
        logits = self.output(x)
        return logits


# Create and run a sample image batch: (2, 1, 32, 32) -> (2, 10).
model = LeNet5()
test_input = torch.randn(2, 1, 32, 32)
logits = model(test_input)

# logits: (2, 10)
