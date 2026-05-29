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


class BasicBlock(nn.Module):
    out_channels: int
    stride: int = 1
    use_projection: bool = False

    @nn.compact
    def __call__(self, x, train=False):
        # Preserve the residual path, projecting it when shape changes.
        identity = x

        # Apply the two-convolution residual branch.
        y = nn.Conv(self.out_channels, (3, 3), strides=(self.stride, self.stride), padding='SAME', use_bias=False, name='conv1')(x)
        y = nn.BatchNorm(use_running_average=not train, name='bn1')(y)
        y = nn.relu(y)
        y = nn.Conv(self.out_channels, (3, 3), padding='SAME', use_bias=False, name='conv2')(y)
        y = nn.BatchNorm(use_running_average=not train, name='bn2')(y)
        if self.use_projection:
            identity = nn.Conv(self.out_channels, (1, 1), strides=(self.stride, self.stride), use_bias=False, name='downsample_conv')(x)
            identity = nn.BatchNorm(use_running_average=not train, name='downsample_bn')(identity)

        # Add residual and apply final activation.
        y = y + identity
        y = nn.relu(y)
        return y


class ResNet18(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Convert image input into stem features.
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding='SAME', use_bias=False, name='stem_conv')(x)
        x = nn.BatchNorm(use_running_average=not train, name='stem_bn')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')

        # Run residual stages while reducing spatial size.
        x = self._stage(x, 64, blocks=2, stride=1, train=train)
        x = self._stage(x, 128, blocks=2, stride=2, train=train)
        x = self._stage(x, 256, blocks=2, stride=2, train=train)
        x = self._stage(x, 512, blocks=2, stride=2, train=train)

        # Pool final features and classify.
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes, name='fc')(x)
        return logits

    def _stage(self, x, channels, blocks, stride, train):
        # Stack residual blocks for one ResNet stage.
        use_projection = stride != 1
        x = BasicBlock(channels, stride, use_projection=use_projection)(x, train=train)
        for _ in range(1, blocks):
            x = BasicBlock(channels)(x, train=train)
        return x


# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
model = ResNet18(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)

# logits: (2, 1000)
