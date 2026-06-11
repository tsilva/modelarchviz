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
class VGG16(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Build stage 1 with two 3x3 convolutions: (batch, 224, 224, 3) -> (batch, 112, 112, 64).
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='conv1_1')(x)  # (batch, 224, 224, 3) -> (batch, 224, 224, 64)
        x = nn.relu(x)  # (batch, 224, 224, 64)
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='conv1_2')(x)  # (batch, 224, 224, 64)
        x = nn.relu(x)  # (batch, 224, 224, 64)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 224, 224, 64) -> (batch, 112, 112, 64)

        # Build stage 2: (batch, 112, 112, 64) -> (batch, 56, 56, 128).
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='conv2_1')(x)  # (batch, 112, 112, 64) -> (batch, 112, 112, 128)
        x = nn.relu(x)  # (batch, 112, 112, 128)
        x = nn.Conv(features=128, kernel_size=(3, 3), padding='SAME', name='conv2_2')(x)  # (batch, 112, 112, 128)
        x = nn.relu(x)  # (batch, 112, 112, 128)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 112, 112, 128) -> (batch, 56, 56, 128)

        # Build stage 3: (batch, 56, 56, 128) -> (batch, 28, 28, 256).
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='conv3_1')(x)  # (batch, 56, 56, 128) -> (batch, 56, 56, 256)
        x = nn.relu(x)  # (batch, 56, 56, 256)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='conv3_2')(x)  # (batch, 56, 56, 256)
        x = nn.relu(x)  # (batch, 56, 56, 256)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='conv3_3')(x)  # (batch, 56, 56, 256)
        x = nn.relu(x)  # (batch, 56, 56, 256)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 56, 56, 256) -> (batch, 28, 28, 256)

        # Build stage 4: (batch, 28, 28, 256) -> (batch, 14, 14, 512).
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='conv4_1')(x)  # (batch, 28, 28, 256) -> (batch, 28, 28, 512)
        x = nn.relu(x)  # (batch, 28, 28, 512)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='conv4_2')(x)  # (batch, 28, 28, 512)
        x = nn.relu(x)  # (batch, 28, 28, 512)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='conv4_3')(x)  # (batch, 28, 28, 512)
        x = nn.relu(x)  # (batch, 28, 28, 512)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 28, 28, 512) -> (batch, 14, 14, 512)

        # Build stage 5: (batch, 14, 14, 512) -> (batch, 7, 7, 512).
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='conv5_1')(x)  # (batch, 14, 14, 512)
        x = nn.relu(x)  # (batch, 14, 14, 512)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='conv5_2')(x)  # (batch, 14, 14, 512)
        x = nn.relu(x)  # (batch, 14, 14, 512)
        x = nn.Conv(features=512, kernel_size=(3, 3), padding='SAME', name='conv5_3')(x)  # (batch, 14, 14, 512)
        x = nn.relu(x)  # (batch, 14, 14, 512)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))  # (batch, 14, 14, 512) -> (batch, 7, 7, 512)

        # Flatten feature maps for the original dense classifier.
        batch_size = x.shape[0]  # (batch, 7, 7, 512) -> scalar
        flat_shape = (batch_size, -1)  # -> (batch, features)
        x = x.reshape(flat_shape)  # (batch, 7, 7, 512) -> (batch, 25088)

        # Classify with two 4096-wide hidden layers.
        x = nn.Dense(features=4096, name='fc6')(x)  # (batch, 25088) -> (batch, 4096)
        x = nn.relu(x)  # (batch, 4096)
        x = nn.Dropout(0.5, deterministic=not train)(x)  # (batch, 4096)
        x = nn.Dense(features=4096, name='fc7')(x)  # (batch, 4096)
        x = nn.relu(x)  # (batch, 4096)
        x = nn.Dropout(0.5, deterministic=not train)(x)  # (batch, 4096)
        logits = nn.Dense(features=self.num_classes, name='fc8')(x)  # (batch, 4096) -> (batch, num_classes)
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
example_model = VGG16(num_classes=1000)
example_test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input, train=False)
example_logits = example_model.apply(example_params, example_test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = VGG16(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images, train=False)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs, train=False)  # (batch, 224, 224, 3) -> (batch, num_classes)
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
