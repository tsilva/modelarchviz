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


class SqueezeExcite(nn.Module):
    squeeze_channels: int

    @nn.compact
    def __call__(self, x):
        # Squeeze spatial dimensions into one descriptor per channel.
        scale = jnp.mean(x, axis=(1, 2), keepdims=True)

        # Excite channels and gate the original feature map.
        scale = nn.Conv(self.squeeze_channels, (1, 1), name='reduce')(scale)
        scale = nn.silu(scale)
        channel_count = x.shape[-1]
        scale = nn.Conv(channel_count, (1, 1), name='expand')(scale)
        scale = nn.sigmoid(scale)
        y = x * scale
        return y


class MBConv(nn.Module):
    out_channels: int
    expand_ratio: int
    stride: int
    kernel_size: int
    se_ratio: float = 0.25

    @nn.compact
    def __call__(self, x, train=False):
        # Configure expanded and squeezed channel widths.
        in_channels = x.shape[-1]
        expanded_channels = in_channels * self.expand_ratio
        squeeze_channels = max(1, int(expanded_channels * self.se_ratio))
        use_residual = self.stride == 1 and in_channels == self.out_channels

        # Expand channels before depthwise spatial filtering.
        y = x
        if self.expand_ratio != 1:
            y = nn.Conv(
                expanded_channels,
                (1, 1),
                use_bias=False,
                name='expand_conv',
            )(y)
            y = nn.BatchNorm(use_running_average=not train, name='expand_bn')(y)
            y = nn.silu(y)

        # Apply one depthwise filter per channel.
        y = nn.Conv(
            expanded_channels,
            (self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding='SAME',
            feature_group_count=expanded_channels,
            use_bias=False,
            name='depthwise_conv',
        )(y)
        y = nn.BatchNorm(use_running_average=not train, name='depthwise_bn')(y)
        y = nn.silu(y)

        # Reweight channels with squeeze-and-excitation, then project back.
        y = SqueezeExcite(squeeze_channels, name='se')(y)
        y = nn.Conv(
            self.out_channels,
            (1, 1),
            use_bias=False,
            name='project_conv',
        )(y)
        y = nn.BatchNorm(use_running_average=not train, name='project_bn')(y)

        # Add the residual shortcut only for same-shape blocks.
        if use_residual:
            y = y + x
        return y


class EfficientNet(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # EfficientNet-B0 stage plan: expand, channels, repeats, stride, kernel.
        settings = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        # Convert image input into stem features: (batch, 224, 224, 3) -> (batch, 112, 112, 32).
        x = nn.Conv(
            32,
            (3, 3),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
            name='stem_conv',
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name='stem_bn')(x)
        x = nn.silu(x)

        # Run compound-scaled MBConv stages with depthwise filters and SE gates.
        block_index = 0
        for expand_ratio, out_channels, repeats, stride, kernel_size in settings:
            for repeat_index in range(repeats):
                block_stride = stride if repeat_index == 0 else 1
                block_name = f'blocks.{block_index}'
                x = MBConv(
                    out_channels,
                    expand_ratio,
                    block_stride,
                    kernel_size,
                    name=block_name,
                )(x, train=train)
                block_index = block_index + 1

        # Expand final channels, pool, and classify.
        x = nn.Conv(
            1280,
            (1, 1),
            use_bias=False,
            name='head_conv',
        )(x)
        x = nn.BatchNorm(use_running_average=not train, name='head_bn')(x)
        x = nn.silu(x)
        x = jnp.mean(x, axis=(1, 2))
        logits = nn.Dense(self.num_classes, name='classifier')(x)
        return logits


# Create and run a sample ImageNet-size batch: (2, 224, 224, 3) -> (2, 1000).
model = EfficientNet(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)

# logits: (2, 1000)

# Train on a tiny synthetic image batch.
model = EfficientNet(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)
train_targets = jnp.array([0, 1])
variables = model.init(jax.random.PRNGKey(1), train_images, train=False)
params = variables['params']
batch_stats = variables['batch_stats']


def train_step(params, batch_stats, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        current_variables = {'params': current_params, 'batch_stats': batch_stats}
        logits = model.apply(current_variables, inputs, train=False)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


for step in range(3):
    params, loss = train_step(params, batch_stats, train_images, train_targets)

final_loss = loss
