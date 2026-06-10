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
        shortcut = x  # (batch, height, width, channels)
        if self.use_projection:
            shortcut = nn.Conv(
                self.out_channels,
                (1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                name='shortcut',
            )(x)  # (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)

        # Run the widened pre-activation residual branch.
        y = nn.BatchNorm(use_running_average=not train, name='bn1')(x)  # (batch, height, width, channels)
        y = nn.relu(y)  # (batch, height, width, channels)
        y = nn.Conv(
            self.out_channels,
            (3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            use_bias=False,
            name='conv1',
        )(y)  # (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name='bn2')(y)  # (batch, out_h, out_w, out_channels)
        y = nn.relu(y)  # (batch, out_h, out_w, out_channels)
        if self.dropout_rate > 0:
            y = nn.Dropout(
                rate=self.dropout_rate,
                name='dropout',
            )(y, deterministic=not train)  # (batch, out_h, out_w, out_channels)
        y = nn.Conv(
            self.out_channels,
            (3, 3),
            padding='SAME',
            use_bias=False,
            name='conv2',
        )(y)  # (batch, out_h, out_w, out_channels)

        # Merge shortcut and residual features.
        y = y + shortcut  # (batch, out_h, out_w, out_channels)
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
        )(x)  # (batch, 32, 32, 3) -> (batch, 32, 32, 16)

        # Run widened residual stages: 160, 320, then 640 channels for WRN-28-10.
        x = self._stage(
            x,
            widths[1],
            block_count,
            stride=1,
            train=train,
            name='layer1',
        )  # (batch, 32, 32, 16) -> (batch, 32, 32, 160)
        x = self._stage(
            x,
            widths[2],
            block_count,
            stride=2,
            train=train,
            name='layer2',
        )  # (batch, 32, 32, 160) -> (batch, 16, 16, 320)
        x = self._stage(
            x,
            widths[3],
            block_count,
            stride=2,
            train=train,
            name='layer3',
        )  # (batch, 16, 16, 320) -> (batch, 8, 8, 640)

        # Pool final feature maps and classify: (batch, 8, 8, 640) -> (batch, 10).
        x = nn.BatchNorm(use_running_average=not train, name='bn')(x)  # (batch, 8, 8, 640)
        x = nn.relu(x)  # (batch, 8, 8, 640)
        x = jnp.mean(x, axis=(1, 2))  # (batch, 8, 8, 640) -> (batch, 640)
        logits = nn.Dense(self.num_classes, name='fc')(x)  # (batch, 640) -> (batch, num_classes)
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
            )(x, train=train)  # (batch, height, width, channels) -> (batch, out_h, out_w, channels)
        return x
