import jax
import jax.numpy as jnp
from flax import linen as nn

class BasicBlock(nn.Module):
    out_channels: int
    stride: int = 1
    use_projection: bool = False
    expansion = 1

    @nn.compact
    def __call__(self, x, train=False):
        # Preserve the residual path, projecting it when shape changes.
        identity = x  # (batch, height, width, in_channels)

        # Apply the two-convolution residual branch.
        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding="SAME", use_bias=False, name="conv1")(x)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn1")(y)  # (batch, out_height, out_width, out_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, out_channels)
        y = nn.Conv(self.out_channels, (3, 3), padding="SAME", use_bias=False, name="conv2")(y)  # (batch, out_height, out_width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn2")(y)  # (batch, out_height, out_width, out_channels)
        if self.use_projection:
            identity = nn.Conv(self.out_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False, name="downsample_conv")(x)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, out_channels)
            identity = nn.BatchNorm(use_running_average=not train, name="downsample_bn")(identity)  # (batch, out_height, out_width, out_channels)

        # Add residual and apply final activation.
        y = y + identity  # (batch, out_height, out_width, out_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, out_channels)
        return y

class Bottleneck(nn.Module):
    out_channels: int
    stride: int = 1
    use_projection: bool = False
    expansion = 4

    @nn.compact
    def __call__(self, x, train=False):
        # Preserve the residual path, projecting it when shape changes.
        identity = x  # (batch, height, width, in_channels)
        expanded_channels = self.out_channels * self.expansion

        # Reduce, process, and expand channels in the residual branch.
        y = nn.Conv(self.out_channels, (1, 1), use_bias=False, name="conv1")(x)  # (batch, height, width, in_channels) -> (batch, height, width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn1")(y)  # (batch, height, width, out_channels)
        y = nn.relu(y)  # (batch, height, width, out_channels)
        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding="SAME", use_bias=False, name="conv2")(y)  # (batch, height, width, out_channels) -> (batch, out_height, out_width, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn2")(y)  # (batch, out_height, out_width, out_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, out_channels)
        y = nn.Conv(expanded_channels, (1, 1), use_bias=False, name="conv3")(y)  # (batch, out_height, out_width, out_channels) -> (batch, out_height, out_width, expanded_channels)
        y = nn.BatchNorm(use_running_average=not train, name="bn3")(y)  # (batch, out_height, out_width, expanded_channels)
        if self.use_projection:
            identity = nn.Conv(expanded_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False, name="downsample_conv")(x)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, expanded_channels)
            identity = nn.BatchNorm(use_running_average=not train, name="downsample_bn")(identity)  # (batch, out_height, out_width, expanded_channels)

        # Add residual and apply final activation.
        y = y + identity  # (batch, out_height, out_width, expanded_channels)
        y = nn.relu(y)  # (batch, out_height, out_width, expanded_channels)
        return y

class ResNet101(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Convert image input into stem features.
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False, name="stem_conv")(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, 64)
        x = nn.BatchNorm(use_running_average=not train, name="stem_bn")(x)  # (batch, 112, 112, 64)
        x = nn.relu(x)  # (batch, 112, 112, 64)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")  # (batch, 112, 112, 64) -> (batch, 56, 56, 64)

        # Run residual stages while reducing spatial size.
        block = Bottleneck
        x = self._stage(x, block, 64, blocks=3, stride=1, train=train)  # (batch, 56, 56, 64)
        x = self._stage(x, block, 128, blocks=4, stride=2, train=train)  # (batch, 56, 56, channels) -> (batch, 28, 28, channels)
        x = self._stage(x, block, 256, blocks=23, stride=2, train=train)  # (batch, 28, 28, channels) -> (batch, 14, 14, channels)
        x = self._stage(x, block, 512, blocks=3, stride=2, train=train)  # (batch, 14, 14, channels) -> (batch, 7, 7, channels)

        # Pool final features and classify.
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, channels) -> (batch, channels)
        logits = nn.Dense(self.num_classes, name="fc")(x)  # (batch, channels) -> (batch, num_classes)
        return logits

    def _stage(self, x, block, channels, blocks, stride, train):
        # Stack residual blocks for one ResNet stage.
        expanded_channels = channels * block.expansion
        use_projection = stride != 1 or x.shape[-1] != expanded_channels
        x = block(channels, stride, use_projection=use_projection)(x, train=train)  # (batch, height, width, in_channels) -> (batch, out_height, out_width, channels)
        for _ in range(1, blocks):
            x = block(channels)(x, train=train)  # (batch, out_height, out_width, channels)
        return x
