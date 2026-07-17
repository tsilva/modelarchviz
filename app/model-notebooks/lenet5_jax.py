# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class LeNet5(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Extract convolutional features: (batch, 32, 32, 1) -> (batch, 5, 5, 16).
        # @arch lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x:start
        conv1 = nn.Conv(features=6, kernel_size=(5, 5))(x)  # (batch, 32, 32, 1) -> (batch, 28, 28, 6)
        # @arch lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x:end
        # @arch lenetn.__call__.x-jnp-tanh-convn:start
        x = jnp.tanh(conv1)  # (batch, 28, 28, 6)
        # @arch lenetn.__call__.x-jnp-tanh-convn:end
        # @arch lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n:start
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 28, 28, 6) -> (batch, 14, 14, 6)
        # @arch lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n:end
        # @arch lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x.2:start
        conv2 = nn.Conv(features=16, kernel_size=(5, 5))(x)  # (batch, 14, 14, 6) -> (batch, 10, 10, 16)
        # @arch lenetn.__call__.convn-nn-conv-features-n-kernel_size-n-n-x.2:end
        # @arch lenetn.__call__.x-jnp-tanh-convn.2:start
        x = jnp.tanh(conv2)  # (batch, 10, 10, 16)
        # @arch lenetn.__call__.x-jnp-tanh-convn.2:end
        # @arch lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n.2:start
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 10, 10, 16) -> (batch, 5, 5, 16)
        # @arch lenetn.__call__.x-nn-avg_pool-x-window_shape-n-n-strides-n-n.2:end

        # Flatten feature maps for dense layers: (batch, 5, 5, 16) -> (batch, 400).
        batch_size = x.shape[0]  # (batch, 5, 5, 16) -> scalar
        flat_shape = (batch_size, -1)  # -> (batch, features)
        # @arch lenetn.__call__.x-x-reshape-flat_shape:start
        x = x.reshape(flat_shape)  # (batch, 5, 5, 16) -> (batch, 400)
        # @arch lenetn.__call__.x-x-reshape-flat_shape:end

        # Classify flattened features: (batch, 400) -> (batch, 10).
        # @arch lenetn.__call__.fcn-nn-dense-features-n-x:start
        fc1 = nn.Dense(features=120)(x)  # (batch, 400) -> (batch, 120)
        # @arch lenetn.__call__.fcn-nn-dense-features-n-x:end
        # @arch lenetn.__call__.x-jnp-tanh-fcn:start
        x = jnp.tanh(fc1)  # (batch, 120)
        # @arch lenetn.__call__.x-jnp-tanh-fcn:end
        # @arch lenetn.__call__.fcn-nn-dense-features-n-x.2:start
        fc2 = nn.Dense(features=84)(x)  # (batch, 120) -> (batch, 84)
        # @arch lenetn.__call__.fcn-nn-dense-features-n-x.2:end
        # @arch lenetn.__call__.x-jnp-tanh-fcn.2:start
        x = jnp.tanh(fc2)  # (batch, 84)
        # @arch lenetn.__call__.x-jnp-tanh-fcn.2:end
        # @arch lenetn.__call__.logits-nn-dense-features-n-x:start
        logits = nn.Dense(features=10)(x)  # (batch, 84) -> (batch, 10)
        # @arch lenetn.__call__.logits-nn-dense-features-n-x:end
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 32, 32, 1) -> (2, 10).
example_model = LeNet5()
example_test_input = jnp.ones((2, 32, 32, 1))  # -> (2, 32, 32, 1)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input)
example_logits = example_model.apply(example_params, example_test_input)  # (2, 32, 32, 1) -> (2, 10)
print("logits shape:", example_logits.shape)

# %%
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
