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
        h1_pre = self.hidden1(x)  # (batch, input_dim) -> (batch, hidden_dim)
        h1 = torch.sigmoid(h1_pre)  # (batch, hidden_dim)

        # Second hidden block keeps the hidden shape: (batch, hidden_dim).
        h2_pre = self.hidden2(h1)  # (batch, hidden_dim)
        h2 = torch.sigmoid(h2_pre)  # (batch, hidden_dim)

        # Output block: (batch, hidden_dim) -> (batch, output_dim).
        logits = self.output(h2)  # (batch, hidden_dim) -> (batch, output_dim)
        return logits
