import jax
import jax.numpy as jnp
from flax import linen as nn

class AlexNet(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Extract early high-stride features: (batch, 227, 227, 3) -> pooled feature maps.
        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4), name='conv1')(x)  # (batch, 227, 227, 3) -> (batch, 55, 55, 96)
        x = nn.relu(x)  # (batch, 55, 55, 96)
        x = local_response_norm(x)  # (batch, 55, 55, 96)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # (batch, 55, 55, 96) -> (batch, 27, 27, 96)

        # Refine convolutional features through middle and late AlexNet blocks.
        x = nn.Conv(features=256, kernel_size=(5, 5), padding='SAME', name='conv2')(x)  # (batch, 27, 27, 96) -> (batch, 27, 27, 256)
        x = nn.relu(x)  # (batch, 27, 27, 256)
        x = local_response_norm(x)  # (batch, 27, 27, 256)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # (batch, 27, 27, 256) -> (batch, 13, 13, 256)
        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv3')(x)  # (batch, 13, 13, 256) -> (batch, 13, 13, 384)
        x = nn.relu(x)  # (batch, 13, 13, 384)
        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv4')(x)  # (batch, 13, 13, 384)
        x = nn.relu(x)  # (batch, 13, 13, 384)
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='conv5')(x)  # (batch, 13, 13, 384) -> (batch, 13, 13, 256)
        x = nn.relu(x)  # (batch, 13, 13, 256)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # (batch, 13, 13, 256) -> (batch, 6, 6, 256)

        # Flatten feature maps for dense classification: (batch, spatial, spatial, channels) -> (batch, features).
        batch_size = x.shape[0]  # (batch, 6, 6, 256) -> scalar
        flat_shape = (batch_size, -1)  # -> (batch, features)
        x = x.reshape(flat_shape)  # (batch, 6, 6, 256) -> (batch, 9216)

        # Classify flattened features: (batch, features) -> (batch, num_classes).
        x = nn.Dropout(0.5, deterministic=not train)(x)  # (batch, 9216)
        x = nn.Dense(features=4096, name='fc6')(x)  # (batch, 9216) -> (batch, 4096)
        x = nn.relu(x)  # (batch, 4096)
        x = nn.Dropout(0.5, deterministic=not train)(x)  # (batch, 4096)
        x = nn.Dense(features=4096, name='fc7')(x)  # (batch, 4096)
        x = nn.relu(x)  # (batch, 4096)
        logits = nn.Dense(features=self.num_classes, name='fc8')(x)  # (batch, 4096) -> (batch, num_classes)
        return logits


def local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0):
    # Build the local channel window used for response normalization.
    half = size // 2
    squared = jnp.square(x)  # (batch, height, width, channels)
    padded = jnp.pad(squared, ((0, 0), (0, 0), (0, 0), (half, half)))  # (batch, height, width, channels) -> (batch, height, width, channels + 2 * half)

    # Accumulate neighboring channel energy and normalize activations.
    scale = k
    for offset in range(size):
        channel_end = offset + x.shape[-1]  # (batch, height, width, channels) -> scalar
        window = padded[..., offset:channel_end]  # (batch, height, width, channels + 2 * half) -> (batch, height, width, channels)
        scale_step = (alpha / size) * window  # (batch, height, width, channels)
        scale = scale + scale_step  # (batch, height, width, channels)
    denominator = jnp.power(scale, beta)  # (batch, height, width, channels)
    normalized = x / denominator  # (batch, height, width, channels)
    return normalized
