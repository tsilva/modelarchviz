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


class DoubleConv(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        # Preserve spatial size while changing channel width.
        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.out_channels, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        return x


class UNet(nn.Module):
    num_classes: int = 2

    @nn.compact
    def __call__(self, x):
        # Encode features while reducing spatial size at each stage.
        d1 = DoubleConv(64)(x)
        p1 = nn.max_pool(d1, (2, 2), (2, 2))
        d2 = DoubleConv(128)(p1)
        p2 = nn.max_pool(d2, (2, 2), (2, 2))
        d3 = DoubleConv(256)(p2)
        p3 = nn.max_pool(d3, (2, 2), (2, 2))
        d4 = DoubleConv(512)(p3)
        p4 = nn.max_pool(d4, (2, 2), (2, 2))

        # Process the bottleneck at the smallest spatial resolution.
        b = DoubleConv(1024)(p4)

        # Decode and concatenate skip features back to full resolution.
        x = resize_like(b, d4)
        x = jnp.concatenate([x, d4], axis=-1)
        x = DoubleConv(512)(x)
        x = resize_like(x, d3)
        x = jnp.concatenate([x, d3], axis=-1)
        x = DoubleConv(256)(x)
        x = resize_like(x, d2)
        x = jnp.concatenate([x, d2], axis=-1)
        x = DoubleConv(128)(x)
        x = resize_like(x, d1)
        x = jnp.concatenate([x, d1], axis=-1)
        x = DoubleConv(64)(x)

        # Project decoder features to segmentation logits.
        logits = nn.Conv(self.num_classes, (1, 1), name='out_conv')(x)
        return logits


def resize_like(x, skip):
    # Resize decoder features to the skip tensor spatial size.
    resize_shape = (x.shape[0], skip.shape[1], skip.shape[2], x.shape[-1])
    resized = jax.image.resize(x, resize_shape, method='nearest')
    return resized


# Create and run a sample image batch: (2, 572, 572, 1) -> (2, 572, 572, 2).
model = UNet(num_classes=2)
test_input = jnp.ones((2, 572, 572, 1))
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)


# Train on two synthetic segmentation masks.
model = UNet(num_classes=2)
train_images = jnp.zeros((2, 64, 64, 1))
train_images = train_images.at[0, 8:32, 8:32, :].set(1.0)
train_images = train_images.at[1, 32:56, 32:56, :].set(1.0)
train_targets = jnp.zeros((2, 64, 64), dtype=jnp.int32)
train_targets = train_targets.at[0, 8:32, 8:32].set(1)
train_targets = train_targets.at[1, 32:56, 32:56].set(1)
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss
