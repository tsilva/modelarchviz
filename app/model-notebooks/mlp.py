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


# Define a multilayer perceptron.
class MLP(nn.Module):
    def __init__(
        self,
        input_dim=784,  # Number of input dimensions.
        hidden_dim=128,  # Number of hidden dimensions.
        output_dim=10  # Number of output dimensions.
    ):
        super().__init__()

        # Register affine projections; initialization does not transform tensors.
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # First hidden block: (batch, input_dim) -> (batch, hidden_dim).
        h1_pre = self.hidden1(x)
        h1 = torch.sigmoid(h1_pre)

        # Second hidden block keeps the hidden shape: (batch, hidden_dim).
        h2_pre = self.hidden2(h1)
        h2 = torch.sigmoid(h2_pre)

        # Output block: (batch, hidden_dim) -> (batch, output_dim).
        logits = self.output(h2)
        return logits


# Create and run a sample batch: (2, 784) -> (2, 10).
model = MLP(input_dim=784, hidden_dim=128, output_dim=10)
inputs = torch.randn(2, 784)
logits = model(inputs)

# Train on a tiny synthetic classification batch.
model = MLP(input_dim=4, hidden_dim=8, output_dim=2)
train_inputs = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
)
train_targets = torch.tensor([0, 1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_inputs)
    loss = criterion(logits, train_targets)
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
