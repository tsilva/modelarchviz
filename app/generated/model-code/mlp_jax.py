import jax
import jax.numpy as jnp
from flax import linen as nn

# Define a multilayer perceptron.
class MLP(nn.Module):
    hidden_dim: int = 128
    output_dim: int = 10

    @nn.compact
    def __call__(self, x):
        # First hidden block: (batch, features) -> (batch, hidden_dim).
        hidden1 = nn.Dense(self.hidden_dim, name='hidden1')
        h1_pre = hidden1(x)  # (batch, features) -> (batch, hidden_dim)
        h1 = nn.sigmoid(h1_pre)  # (batch, hidden_dim)

        # Second hidden block keeps the hidden shape: (batch, hidden_dim).
        hidden2 = nn.Dense(self.hidden_dim, name='hidden2')
        h2_pre = hidden2(h1)  # (batch, hidden_dim)
        h2 = nn.sigmoid(h2_pre)  # (batch, hidden_dim)

        # Output block: (batch, hidden_dim) -> (batch, output_dim).
        output = nn.Dense(self.output_dim, name='output')
        logits = output(h2)  # (batch, hidden_dim) -> (batch, output_dim)
        return logits
