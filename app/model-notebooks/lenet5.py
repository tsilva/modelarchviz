# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
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


# %% [notebook-only]
# Create and run a sample image batch: (2, 1, 32, 32) -> (2, 10).
model = LeNet5()
test_input = torch.randn(2, 1, 32, 32)  # -> (2, 1, 32, 32)
logits = model(test_input)  # (2, 1, 32, 32) -> (2, 10)


# Train on a tiny synthetic image batch.
model = LeNet5()
train_images = torch.zeros(2, 1, 32, 32)  # -> (2, 1, 32, 32)
train_images[0, :, 8:16, 8:16] = 1.0
train_images[1, :, 16:24, 16:24] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 1, 32, 32) -> (2, 10)
    loss = criterion(logits, train_targets)  # (2, 10), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
