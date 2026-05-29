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

# logits: (2, 572, 572, 2)
