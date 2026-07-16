# %%
import torch
import torch.nn as nn


# %%
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
        h1_pre = self.hidden1(x)  # (batch, input_dim) -> (batch, hidden_dim)
        h1 = torch.sigmoid(h1_pre)  # (batch, hidden_dim)

        # Second hidden block keeps the hidden shape: (batch, hidden_dim).
        h2_pre = self.hidden2(h1)  # (batch, hidden_dim)
        h2 = torch.sigmoid(h2_pre)  # (batch, hidden_dim)

        # Output block: (batch, hidden_dim) -> (batch, output_dim).
        logits = self.output(h2)  # (batch, hidden_dim) -> (batch, output_dim)
        return logits


# %% [notebook-only]
# Create and run a sample batch: (2, 784) -> (2, 10).
example_model = MLP(input_dim=784, hidden_dim=128, output_dim=10)
inputs = torch.randn(2, 784)  # -> (2, 784)
example_logits = example_model(inputs)  # (2, 784) -> (2, 10)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic classification batch.
model = MLP(input_dim=4, hidden_dim=8, output_dim=2)
train_inputs = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
)  # -> (2, 4)
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_inputs)  # (2, 4) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
