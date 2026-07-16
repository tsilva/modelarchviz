# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class SqueezeExcite(nn.Module):
    squeeze_channels: int

    @nn.compact
    def __call__(self, x):
        # Squeeze spatial dimensions into one descriptor per channel.
        scale = jnp.mean(x, axis=(1, 2), keepdims=True)  # (batch, height, width, channels) -> (batch, 1, 1, channels)

        # Excite channels and gate the original feature map.
        scale = nn.Conv(self.squeeze_channels, (1, 1), name='reduce')(scale)  # (batch, 1, 1, channels) -> (batch, 1, 1, squeeze_channels)
        scale = nn.silu(scale)  # (batch, 1, 1, squeeze_channels)
        channel_count = x.shape[-1]  # (batch, height, width, channels) -> scalar
        scale = nn.Conv(channel_count, (1, 1), name='expand')(scale)  # (batch, 1, 1, squeeze_channels) -> (batch, 1, 1, channels)
        scale = nn.sigmoid(scale)  # (batch, 1, 1, channels)
        y = x * scale  # (batch, height, width, channels)
        return y


# %% [notebook-only]
# Create and run a squeeze-excitation gate: (2, 16, 16, 8) -> (2, 16, 16, 8).
gate = SqueezeExcite(squeeze_channels=2)
feature_map = jnp.ones((2, 16, 16, 8))  # -> (2, 16, 16, 8)
example_params = gate.init(jax.random.PRNGKey(0), feature_map)
gated = gate.apply(example_params, feature_map)  # (2, 16, 16, 8) -> (2, 16, 16, 8)
print("gated shape:", gated.shape)

# %%
class MBConv(nn.Module):
    out_channels: int
    expand_ratio: int
    stride: int
    kernel_size: int
    se_ratio: float = 0.25

    @nn.compact
    def __call__(self, x, train=False):
        # Configure expanded and squeezed channel widths.
        in_channels = x.shape[-1]  # (batch, height, width, in_channels) -> scalar
        expanded_channels = in_channels * self.expand_ratio
        squeeze_channels = max(1, int(expanded_channels * self.se_ratio))
        use_residual = self.stride == 1 and in_channels == self.out_channels

        # Expand channels before depthwise spatial filtering.
        y = x  # (batch, height, width, in_channels)
        if self.expand_ratio != 1:
            y = nn.Conv(
                expanded_channels,
                (1, 1),
                use_bias=False,
                name='expand_conv',
            )(y)  # (batch, height, width, in_channels) -> (batch, height, width, expanded_channels)
            y = nn.BatchNorm(use_running_average=not train, name='expand_bn')(y)  # (batch, height, width, expanded_channels)
            y = nn.silu(y)  # (batch, height, width, expanded_channels)

        # Apply one depthwise filter per channel.
        y = nn.Conv(
            expanded_channels,
            (self.kernel_size, self.kernel_size),
            strides=(self.stride, self.stride),
            padding='SAME',
            feature_group_count=expanded_channels,
            use_bias=False,
            name='depthwise_conv',
        )(y)  # (batch, height, width, expanded_channels) -> (batch, out_h, out_w, expanded_channels)
        y = nn.BatchNorm(use_running_average=not train, name='depthwise_bn')(y)  # (batch, out_h, out_w, expanded_channels)
        y = nn.silu(y)  # (batch, out_h, out_w, expanded_channels)

        # Reweight channels with squeeze-and-excitation, then project back.
        y = SqueezeExcite(squeeze_channels, name='se')(y)  # (batch, out_h, out_w, expanded_channels)
        y = nn.Conv(
            self.out_channels,
            (1, 1),
            use_bias=False,
            name='project_conv',
        )(y)  # (batch, out_h, out_w, expanded_channels) -> (batch, out_h, out_w, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name='project_bn')(y)  # (batch, out_h, out_w, out_channels)

        # Add the residual shortcut only for same-shape blocks.
        if use_residual:
            y = y + x  # (batch, height, width, out_channels)
        return y


# %% [notebook-only]
# Create and run one mobile inverted bottleneck: (2, 16, 16, 8) -> (2, 16, 16, 8).
example_block = MBConv(out_channels=8, expand_ratio=1, stride=1, kernel_size=3)
block_input = jnp.ones((2, 16, 16, 8))  # -> (2, 16, 16, 8)
variables = example_block.init(jax.random.PRNGKey(1), block_input, train=False)
example_block_output = example_block.apply(variables, block_input, train=False)  # (2, 16, 16, 8) -> (2, 16, 16, 8)
print("block_output shape:", example_block_output.shape)

# %%
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
        )(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, 32)
        x = nn.BatchNorm(use_running_average=not train, name='stem_bn')(x)  # (batch, 112, 112, 32)
        x = nn.silu(x)  # (batch, 112, 112, 32)

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
                )(x, train=train)  # (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)
                block_index = block_index + 1

        # Expand final channels, pool, and classify.
        x = nn.Conv(
            1280,
            (1, 1),
            use_bias=False,
            name='head_conv',
        )(x)  # (batch, 7, 7, 320) -> (batch, 7, 7, 1280)
        x = nn.BatchNorm(use_running_average=not train, name='head_bn')(x)  # (batch, 7, 7, 1280)
        x = nn.silu(x)  # (batch, 7, 7, 1280)
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, 1280) -> (batch, 1280)
        logits = nn.Dense(self.num_classes, name='classifier')(x)  # (batch, 1280) -> (batch, num_classes)
        return logits


# %% [notebook-only]
# Create and run a sample ImageNet-size batch: (2, 224, 224, 3) -> (2, 1000).
example_model = EfficientNet(num_classes=1000)
example_test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input, train=False)
example_logits = example_model.apply(example_params, example_test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = EfficientNet(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
variables = model.init(jax.random.PRNGKey(1), train_images, train=False)
params = variables['params']
batch_stats = variables['batch_stats']


def train_step(params, batch_stats, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        current_variables = {'params': current_params, 'batch_stats': batch_stats}
        logits = model.apply(current_variables, inputs, train=False)  # (batch, 224, 224, 3) -> (batch, num_classes)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, num_classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, num_classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, num_classes) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, batch_stats, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
