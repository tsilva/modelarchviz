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


class WideBasicBlock(nn.Module):
    out_channels: int
    stride: int = 1
    dropout_rate: float = 0.0
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        # Preserve the shortcut path, projecting it when width or spatial size changes.
        shortcut = x
        if self.use_projection:
            shortcut = nn.Conv(
                self.out_channels,
                (1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                name='shortcut',
            )(x)

        # Run the widened pre-activation residual branch.
        y = nn.BatchNorm(use_running_average=not train, name='bn1')(x)
        y = nn.relu(y)
        y = nn.Conv(
            self.out_channels,
            (3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            use_bias=False,
            name='conv1',
        )(y)
        y = nn.BatchNorm(use_running_average=not train, name='bn2')(y)
        y = nn.relu(y)
        if self.dropout_rate > 0:
            y = nn.Dropout(
                rate=self.dropout_rate,
                name='dropout',
            )(y, deterministic=not train)
        y = nn.Conv(
            self.out_channels,
            (3, 3),
            padding='SAME',
            use_bias=False,
            name='conv2',
        )(y)

        # Merge shortcut and residual features.
        y = y + shortcut
        return y


class WideNet(nn.Module):
    depth: int = 28
    widen_factor: int = 10
    dropout_rate: float = 0.0
    num_classes: int = 10

    @nn.compact
    def __call__(self, x, train=False):
        # Configure the WRN-28-10 channel plan for CIFAR-size inputs.
        if (self.depth - 4) % 6 != 0:
            raise ValueError("WideNet depth must satisfy depth = 6n + 4.")
        block_count = (self.depth - 4) // 6
        widths = [
            16,
            16 * self.widen_factor,
            32 * self.widen_factor,
            64 * self.widen_factor,
        ]

        # Convert image input into low-level features: (batch, 32, 32, 3) -> (batch, 32, 32, 16).
        x = nn.Conv(
            widths[0],
            (3, 3),
            padding='SAME',
            use_bias=False,
            name='conv1',
        )(x)

        # Run widened residual stages: 160, 320, then 640 channels for WRN-28-10.
        x = self._stage(
            x,
            widths[1],
            block_count,
            stride=1,
            train=train,
            name='layer1',
        )
        x = self._stage(
            x,
            widths[2],
            block_count,
            stride=2,
            train=train,
            name='layer2',
        )
        x = self._stage(
            x,
            widths[3],
            block_count,
            stride=2,
            train=train,
            name='layer3',
        )

        # Pool final feature maps and classify: (batch, 8, 8, 640) -> (batch, 10).
        x = nn.BatchNorm(use_running_average=not train, name='bn')(x)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes, name='fc')(x)
        return logits

    def _stage(self, x, channels, blocks, stride, train, name):
        # Start each stage with the only block that may widen channels or downsample.
        for index in range(blocks):
            block_stride = stride if index == 0 else 1
            use_projection = index == 0
            block_name = f'{name}.{index}'
            x = WideBasicBlock(
                channels,
                stride=block_stride,
                dropout_rate=self.dropout_rate,
                use_projection=use_projection,
                name=block_name,
            )(x, train=train)
        return x


# Create and run a sample CIFAR-size image batch: (2, 32, 32, 3) -> (2, 10).
model = WideNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=10)
test_input = jnp.ones((2, 32, 32, 3))
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)

# logits: (2, 10)
