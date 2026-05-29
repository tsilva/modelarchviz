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
import jax
import jax.numpy as jnp
from flax import linen as nn


class LeNet5(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Extract convolutional features: (batch, 32, 32, 1) -> (batch, 5, 5, 16).
        conv1 = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = jnp.tanh(conv1)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        conv2 = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = jnp.tanh(conv2)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten feature maps for dense layers: (batch, 5, 5, 16) -> (batch, 400).
        batch_size = x.shape[0]
        flat_shape = (batch_size, -1)
        x = x.reshape(flat_shape)

        # Classify flattened features: (batch, 400) -> (batch, 10).
        fc1 = nn.Dense(features=120)(x)
        x = jnp.tanh(fc1)
        fc2 = nn.Dense(features=84)(x)
        x = jnp.tanh(fc2)
        logits = nn.Dense(features=10)(x)
        return logits


# Create and run a sample image batch: (2, 32, 32, 1) -> (2, 10).
model = LeNet5()
test_input = jnp.ones((2, 32, 32, 1))
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)

# logits: (2, 10)
