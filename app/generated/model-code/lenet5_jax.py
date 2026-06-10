import jax
import jax.numpy as jnp
from flax import linen as nn

class LeNet5(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Extract convolutional features: (batch, 32, 32, 1) -> (batch, 5, 5, 16).
        conv1 = nn.Conv(features=6, kernel_size=(5, 5))(x)  # (batch, 32, 32, 1) -> (batch, 28, 28, 6)
        x = jnp.tanh(conv1)  # (batch, 28, 28, 6)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 28, 28, 6) -> (batch, 14, 14, 6)
        conv2 = nn.Conv(features=16, kernel_size=(5, 5))(x)  # (batch, 14, 14, 6) -> (batch, 10, 10, 16)
        x = jnp.tanh(conv2)  # (batch, 10, 10, 16)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 10, 10, 16) -> (batch, 5, 5, 16)

        # Flatten feature maps for dense layers: (batch, 5, 5, 16) -> (batch, 400).
        batch_size = x.shape[0]  # (batch, 5, 5, 16) -> scalar
        flat_shape = (batch_size, -1)  # -> (batch, features)
        x = x.reshape(flat_shape)  # (batch, 5, 5, 16) -> (batch, 400)

        # Classify flattened features: (batch, 400) -> (batch, 10).
        fc1 = nn.Dense(features=120)(x)  # (batch, 400) -> (batch, 120)
        x = jnp.tanh(fc1)  # (batch, 120)
        fc2 = nn.Dense(features=84)(x)  # (batch, 120) -> (batch, 84)
        x = jnp.tanh(fc2)  # (batch, 84)
        logits = nn.Dense(features=10)(x)  # (batch, 84) -> (batch, 10)
        return logits
