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


class DenseLayer(nn.Module):
    growth_rate: int
    bottleneck_width: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train=False):
        # Compress existing features, then produce new growth features.
        bottleneck_channels = self.bottleneck_width * self.growth_rate
        y = nn.BatchNorm(use_running_average=not train, name='norm1')(x)
        y = nn.relu(y)
        y = nn.Conv(
            bottleneck_channels,
            (1, 1),
            use_bias=False,
            name='conv1',
        )(y)
        y = nn.BatchNorm(use_running_average=not train, name='norm2')(y)
        y = nn.relu(y)
        y = nn.Conv(
            self.growth_rate,
            (3, 3),
            padding='SAME',
            use_bias=False,
            name='conv2',
        )(y)
        if self.dropout_rate > 0:
            y = nn.Dropout(
                rate=self.dropout_rate,
                name='dropout',
            )(y, deterministic=not train)

        # Concatenate old and new features: (batch, height, width, channels) grows by growth_rate.
        features = [x, y]
        y = jnp.concatenate(features, axis=-1)
        return y


class Transition(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, train=False):
        # Compress channels and halve spatial resolution.
        y = nn.BatchNorm(use_running_average=not train, name='norm')(x)
        y = nn.relu(y)
        y = nn.Conv(
            self.out_channels,
            (1, 1),
            use_bias=False,
            name='conv',
        )(y)
        y = nn.avg_pool(y, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        return y


class DenseNet(nn.Module):
    growth_rate: int = 32
    block_config: tuple = (6, 12, 24, 16)
    num_init_features: int = 64
    compression: float = 0.5
    dropout_rate: float = 0.0
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Convert image input into stem features: (batch, 224, 224, 3) -> (batch, 56, 56, 64).
        x = nn.Conv(
            self.num_init_features,
            (7, 7),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
            name='stem_conv',
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name='stem_norm')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Grow and compress feature maps through dense blocks and transitions.
        num_features = self.num_init_features
        for block_index, layer_count in enumerate(self.block_config):
            for layer_index in range(layer_count):
                layer_name = f'denseblock{block_index + 1}.layer{layer_index + 1}'
                x = DenseLayer(
                    self.growth_rate,
                    dropout_rate=self.dropout_rate,
                    name=layer_name,
                )(x, train=train)
                num_features = num_features + self.growth_rate

            is_last_block = block_index == len(self.block_config) - 1
            if not is_last_block:
                num_features = int(num_features * self.compression)
                transition_name = f'transition{block_index + 1}'
                x = Transition(num_features, name=transition_name)(x, train=train)

        # Normalize, pool, and classify final dense features.
        x = nn.BatchNorm(use_running_average=not train, name='norm')(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes, name='classifier')(x)
        return logits


# Create and run a sample ImageNet-size batch: (2, 224, 224, 3) -> (2, 1000).
model = DenseNet(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)

# logits: (2, 1000)
