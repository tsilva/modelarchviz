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


# %%
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


# %% [notebook-only]
# Create and run a sample image batch: (2, 32, 32, 1) -> (2, 10).
model = LeNet5()
test_input = jnp.ones((2, 32, 32, 1))  # -> (2, 32, 32, 1)
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)  # (2, 32, 32, 1) -> (2, 10)


# Train on a tiny synthetic image batch.
model = LeNet5()
train_images = jnp.zeros((2, 32, 32, 1))  # -> (2, 32, 32, 1)
train_images = train_images.at[0, 8:16, 8:16, :].set(1.0)  # (2, 32, 32, 1)
train_images = train_images.at[1, 16:24, 16:24, :].set(1.0)  # (2, 32, 32, 1)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # (batch, 32, 32, 1) -> (batch, 10)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, classes) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss
