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
        branch1 = nn.Conv(self.branch1_channels, (1, 1), name='branch1')(x)
        branch1 = nn.relu(branch1)

        branch3 = nn.Conv(self.branch3_reduce, (1, 1), name='branch3_reduce')(x)
        branch3 = nn.relu(branch3)
        branch3 = nn.Conv(self.branch3_channels, (3, 3), padding='SAME', name='branch3')(branch3)
        branch3 = nn.relu(branch3)

        branch5 = nn.Conv(self.branch5_reduce, (1, 1), name='branch5_reduce')(x)
        branch5 = nn.relu(branch5)
        branch5 = nn.Conv(self.branch5_channels, (5, 5), padding='SAME', name='branch5')(branch5)
        branch5 = nn.relu(branch5)

        branch_pool = nn.max_pool(x, window_shape=(3, 3), strides=(1, 1), padding='SAME')
        branch_pool = nn.Conv(self.pool_channels, (1, 1), name='pool_proj')(branch_pool)
        branch_pool = nn.relu(branch_pool)

        # Concatenate branch channels: list of (batch, height, width, channels) -> one feature map.
        branches = [branch1, branch3, branch5, branch_pool]
        x = jnp.concatenate(branches, axis=-1)
        return x


class GoogLeNet(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Downsample the input into stem features: (batch, 224, 224, 3) -> stem feature maps.
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME', name='stem_conv7')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = nn.Conv(64, (1, 1), name='stem_conv1')(x)
        x = nn.relu(x)
        x = nn.Conv(192, (3, 3), padding='SAME', name='stem_conv3')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Run Inception stage 3 and downsample spatial size.
        x = InceptionBlock(64, 96, 128, 16, 32, 32, name='inception3a')(x)
        x = InceptionBlock(128, 128, 192, 32, 96, 64, name='inception3b')(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Run Inception stage 4 and downsample spatial size.
        x = InceptionBlock(192, 96, 208, 16, 48, 64, name='inception4a')(x)
        x = InceptionBlock(160, 112, 224, 24, 64, 64, name='inception4b')(x)
        x = InceptionBlock(128, 128, 256, 24, 64, 64, name='inception4c')(x)
        x = InceptionBlock(112, 144, 288, 32, 64, 64, name='inception4d')(x)
        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception4e')(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Run Inception stage 5 and pool to a classifier vector.
        x = InceptionBlock(256, 160, 320, 32, 128, 128, name='inception5a')(x)
        x = InceptionBlock(384, 192, 384, 48, 128, 128, name='inception5b')(x)

        # Apply dropout and classify pooled features: (batch, 1024) -> (batch, num_classes).
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dropout(0.4, deterministic=not train)(x)
        logits = nn.Dense(self.num_classes, name='fc')(x)
        return logits


# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
model = GoogLeNet(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)

# logits: (2, 1000)
