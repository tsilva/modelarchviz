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


class InceptionBlock(nn.Module):
    branch1_channels: int
    branch3_reduce: int
    branch3_channels: int
    branch5_reduce: int
    branch5_channels: int
    pool_channels: int

    @nn.compact
    def __call__(self, x):
        # Evaluate parallel 1x1 and larger-kernel branches.
        branch1 = nn.Conv(self.branch1_channels, (1, 1), name='branch1')(x)  # (batch, height, width, in_channels) -> (batch, height, width, branch1_channels)
        branch1 = nn.relu(branch1)  # (batch, height, width, branch1_channels)

        branch3 = nn.Conv(self.branch3_reduce, (1, 1), name='branch3_reduce')(x)  # (batch, height, width, in_channels) -> (batch, height, width, branch3_reduce)
        branch3 = nn.relu(branch3)  # (batch, height, width, branch3_reduce)
        branch3 = nn.Conv(self.branch3_channels, (3, 3), padding='SAME', name='branch3')(branch3)  # (batch, height, width, branch3_reduce) -> (batch, height, width, branch3_channels)
        branch3 = nn.relu(branch3)  # (batch, height, width, branch3_channels)

        branch5 = nn.Conv(self.branch5_reduce, (1, 1), name='branch5_reduce')(x)  # (batch, height, width, in_channels) -> (batch, height, width, branch5_reduce)
        branch5 = nn.relu(branch5)  # (batch, height, width, branch5_reduce)
        branch5 = nn.Conv(self.branch5_channels, (5, 5), padding='SAME', name='branch5')(branch5)  # (batch, height, width, branch5_reduce) -> (batch, height, width, branch5_channels)
        branch5 = nn.relu(branch5)  # (batch, height, width, branch5_channels)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')  # (batch, height, width, in_channels)
        branch_pool = nn.Conv(self.pool_channels, (1, 1), name='pool_proj')(branch_pool)  # (batch, height, width, in_channels) -> (batch, height, width, pool_channels)
        branch_pool = nn.relu(branch_pool)  # (batch, height, width, pool_channels)

        # Concatenate branch channels: list of (batch, height, width, channels) -> one feature map.
        branches = [branch1, branch3, branch5, branch_pool]
        x = jnp.concatenate(branches, axis=-1)  # list of (batch, height, width, channels) -> (batch, height, width, output_channels)
        return x


class GoogLeNet(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Downsample the input into stem features: (batch, 224, 224, 3) -> stem feature maps.
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME', name='stem_conv7')(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, 64)
        x = nn.relu(x)  # (batch, 112, 112, 64)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 112, 112, 64) -> (batch, 56, 56, 64)
        x = nn.Conv(64, (1, 1), name='stem_conv1')(x)  # (batch, 56, 56, 64)
        x = nn.relu(x)  # (batch, 56, 56, 64)
        x = nn.Conv(192, (3, 3), padding='SAME', name='stem_conv3')(x)  # (batch, 56, 56, 64) -> (batch, 56, 56, 192)
        x = nn.relu(x)  # (batch, 56, 56, 192)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 56, 56, 192) -> (batch, 28, 28, 192)

        # Run Inception stage 3 and downsample spatial size.
        x = InceptionBlock(64, 96, 128, 16, 32, 32, name='inception3a')(x)  # (batch, 28, 28, 192) -> (batch, 28, 28, 256)
        x = InceptionBlock(128, 128, 192, 32, 96, 64, name='inception3b')(x)  # (batch, 28, 28, 256) -> (batch, 28, 28, 480)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 28, 28, 480) -> (batch, 14, 14, 480)

        # Run Inception stage 4 and downsample spatial size.
        x = InceptionBlock(192, 96, 208, 16, 48, 64, name='inception4a')(x)  # (batch, 14, 14, 480) -> (batch, 14, 14, 512)
        x = InceptionBlock(160, 112, 224, 24, 64, 64, name='inception4b')(x)  # (batch, 14, 14, 512)
        x = InceptionBlock(128, 128, 256, 24, 64, 64, name='inception4c')(x)  # (batch, 14, 14, 512)
        x = InceptionBlock(112, 144, 288, 32, 64, 64, name='inception4d')(x)  # (batch, 14, 14, 512) -> (batch, 14, 14, 528)
        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception4e')(x)  # (batch, 14, 14, 528) -> (batch, 14, 14, 832)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 14, 14, 832) -> (batch, 7, 7, 832)

        # Run Inception stage 5 and pool to a classifier vector.
        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception5a')(x)  # (batch, 7, 7, 832)
        x = InceptionBlock(384, 192, 384, 48, 128, 128, name='inception5b')(x)  # (batch, 7, 7, 832) -> (batch, 7, 7, 1024)

        # Apply dropout and classify pooled features: (batch, 1024) -> (batch, num_classes).
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, 1024) -> (batch, 1024)
        x = nn.Dropout(0.4, deterministic=not train)(x)  # (batch, 1024)
        logits = nn.Dense(self.num_classes, name='fc')(x)  # (batch, 1024) -> (batch, num_classes)
        return logits


# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
model = GoogLeNet(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)


# Train on a tiny synthetic image batch.
model = GoogLeNet(num_classes=2)
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
