# %%
import torch
import torch.nn as nn


# %%
# Define a multilayer perceptron.
class MLP(nn.Module):
    def __init__(
        self,
        # @arch mlp.input_dim-n:start
        input_dim=784,  # Number of input dimensions.
        # @arch mlp.input_dim-n:end
        # @arch mlp.hidden_dim-n:start
        hidden_dim=128,  # Number of hidden dimensions.
        # @arch mlp.hidden_dim-n:end
        # @arch mlp.output_dim-n:start
        output_dim=10  # Number of output dimensions.
        # @arch mlp.output_dim-n:end
    ):
        super().__init__()

        # Register affine projections; initialization does not transform tensors.
        # @arch mlp.self-hiddenn-nn-linear-input_dim-hidden_dim:start
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        # @arch mlp.self-hiddenn-nn-linear-input_dim-hidden_dim:end
        # @arch mlp.self-hiddenn-nn-linear-hidden_dim-hidden_dim:start
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        # @arch mlp.self-hiddenn-nn-linear-hidden_dim-hidden_dim:end
        # @arch mlp.self-output-nn-linear-hidden_dim-output_dim:start
        self.output = nn.Linear(hidden_dim, output_dim)
        # @arch mlp.self-output-nn-linear-hidden_dim-output_dim:end

    def forward(self, x):
        # First hidden block: (batch, input_dim) -> (batch, hidden_dim).
        # @arch mlp.forward.hn_pre-self-hiddenn-x:start
        h1_pre = self.hidden1(x)  # (batch, input_dim) -> (batch, hidden_dim)
        # @arch mlp.forward.hn_pre-self-hiddenn-x:end
        # @arch mlp.forward.hn-torch-sigmoid-hn_pre:start
        h1 = torch.sigmoid(h1_pre)  # (batch, hidden_dim)
        # @arch mlp.forward.hn-torch-sigmoid-hn_pre:end

        # Second hidden block keeps the hidden shape: (batch, hidden_dim).
        # @arch mlp.forward.hn_pre-self-hiddenn-hn:start
        h2_pre = self.hidden2(h1)  # (batch, hidden_dim)
        # @arch mlp.forward.hn_pre-self-hiddenn-hn:end
        # @arch mlp.forward.hn-torch-sigmoid-hn_pre.2:start
        h2 = torch.sigmoid(h2_pre)  # (batch, hidden_dim)
        # @arch mlp.forward.hn-torch-sigmoid-hn_pre.2:end

        # Output block: (batch, hidden_dim) -> (batch, output_dim).
        # @arch mlp.forward.logits-self-output-hn:start
        logits = self.output(h2)  # (batch, hidden_dim) -> (batch, output_dim)
        # @arch mlp.forward.logits-self-output-hn:end
        # @arch mlp.forward.return-logits:start
        return logits
        # @arch mlp.forward.return-logits:end


# %% [notebook-only]
# Create and run a sample batch: (2, 784) -> (2, 10).
example_model = MLP(input_dim=784, hidden_dim=128, output_dim=10)
inputs = torch.randn(2, 784)  # -> (2, 784)
example_logits = example_model(inputs)  # (2, 784) -> (2, 10)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic classification batch.
model = MLP(input_dim=4, hidden_dim=8, output_dim=2)
# @arch train_inputs-torch-tensor:start
train_inputs = torch.tensor(
# @arch train_inputs-torch-tensor:end
    # @arch code.4:start
    [
    # @arch code.4:end
        # @arch n-n-n-n:start
        [1.0, 0.0, 1.0, 0.0],
        # @arch n-n-n-n:end
        # @arch n-n-n-n.2:start
        [0.0, 1.0, 0.0, 1.0],
        # @arch n-n-n-n.2:end
    # @arch code.5:start
    ]
    # @arch code.5:end
# @arch code.6:start
)  # -> (2, 4)
# @arch code.6:end
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    # @arch logits-model-train_inputs:start
    logits = model(train_inputs)  # (2, 4) -> (2, 2)
    # @arch logits-model-train_inputs:end
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
