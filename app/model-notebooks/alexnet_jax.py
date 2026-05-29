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


class AlexNet(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Extract early high-stride features: (batch, 227, 227, 3) -> pooled feature maps.
        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4), name='conv1')(x)
        x = nn.relu(x)
        x = local_response_norm(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Refine convolutional features through middle and late AlexNet blocks.
        x = nn.Conv(features=256, kernel_size=(5, 5), padding='SAME', name='conv2')(x)
        x = nn.relu(x)
        x = local_response_norm(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv3')(x)
        x = nn.relu(x)
        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv4')(x)
        x = nn.relu(x)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='conv5')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        # Flatten feature maps for dense classification: (batch, spatial, spatial, channels) -> (batch, features).
        batch_size = x.shape[0]
        flat_shape = (batch_size, -1)
        x = x.reshape(flat_shape)

        # Classify flattened features: (batch, features) -> (batch, num_classes).
        x = nn.Dropout(0.5, deterministic=not train)(x)
        x = nn.Dense(features=4096, name='fc6')(x)
        x = nn.relu(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)
        x = nn.Dense(features=4096, name='fc7')(x)
        x = nn.relu(x)
        logits = nn.Dense(features=self.num_classes, name='fc8')(x)
        return logits


def local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0):
    # Build the local channel window used for response normalization.
    half = size // 2
    squared = jnp.square(x)
    padded = jnp.pad(squared, ((0, 0), (0, 0), (0, 0), (half, half)))

    # Accumulate neighboring channel energy and normalize activations.
    scale = k
    for offset in range(size):
        channel_end = offset + x.shape[-1]
        window = padded[..., offset:channel_end]
        scale_step = (alpha / size) * window
        scale = scale + scale_step
    denominator = jnp.power(scale, beta)
    normalized = x / denominator
    return normalized


# Create and run a sample image batch: (2, 227, 227, 3) -> (2, 1000).
model = AlexNet(num_classes=1000)
test_input = jnp.ones((2, 227, 227, 3))
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)

# logits: (2, 1000)
