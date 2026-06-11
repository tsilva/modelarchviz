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


# %%
class InvertedResidual(nn.Module):
    out_channels: int
    stride: int
    expand_ratio: int

    @nn.compact
    def __call__(self, x, train=False):
        # Configure expanded hidden width and residual eligibility.
        in_channels = x.shape[-1]  # (batch, height, width, in_channels) -> scalar
        hidden_channels = in_channels * self.expand_ratio
        use_residual = self.stride == 1 and in_channels == self.out_channels

        # Expand narrow input bottlenecks when the expansion ratio is greater than one.
        y = x  # (batch, height, width, in_channels)
        if self.expand_ratio != 1:
            y = nn.Conv(
                hidden_channels,
                (1, 1),
                use_bias=False,
                name='expand_conv',
            )(y)  # (batch, height, width, in_channels) -> (batch, height, width, hidden_channels)
            y = nn.BatchNorm(use_running_average=not train, name='expand_bn')(y)  # (batch, height, width, hidden_channels)
            y = nn.relu6(y)  # (batch, height, width, hidden_channels)

        # Apply one spatial 3x3 filter per expanded channel.
        y = nn.Conv(
            hidden_channels,
            (3, 3),
            strides=(self.stride, self.stride),
            padding='SAME',
            feature_group_count=hidden_channels,
            use_bias=False,
            name='depthwise_conv',
        )(y)  # (batch, height, width, hidden_channels) -> (batch, out_h, out_w, hidden_channels)
        y = nn.BatchNorm(use_running_average=not train, name='depthwise_bn')(y)  # (batch, out_h, out_w, hidden_channels)
        y = nn.relu6(y)  # (batch, out_h, out_w, hidden_channels)

        # Project back to a narrow linear bottleneck without an activation.
        y = nn.Conv(
            self.out_channels,
            (1, 1),
            use_bias=False,
            name='project_conv',
        )(y)  # (batch, out_h, out_w, hidden_channels) -> (batch, out_h, out_w, out_channels)
        y = nn.BatchNorm(use_running_average=not train, name='project_bn')(y)  # (batch, out_h, out_w, out_channels)

        # Add the shortcut only when bottleneck shapes match.
        if use_residual:
            y = y + x  # (batch, height, width, out_channels)
        return y


# %% [notebook-only]
# Create and run one inverted residual block: (2, 32, 32, 16) -> (2, 32, 32, 16).
example_block = InvertedResidual(out_channels=16, stride=1, expand_ratio=6)
example_block_input = jnp.ones((2, 32, 32, 16))  # -> (2, 32, 32, 16)
example_variables = example_block.init(jax.random.PRNGKey(0), example_block_input, train=False)
example_block_output = example_block.apply(example_variables, example_block_input, train=False)  # (2, 32, 32, 16) -> (2, 32, 32, 16)
print("block_output shape:", example_block_output.shape)

# %%
class MobileNetV2(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # MobileNetV2 stage plan: expansion, output channels, repeats, first stride.
        settings = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
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
        x = nn.relu6(x)  # (batch, 112, 112, 32)

        # Run inverted residual stages with depthwise filters and linear bottlenecks.
        block_index = 0
        for expand_ratio, out_channels, repeats, stride in settings:
            for repeat_index in range(repeats):
                block_stride = stride if repeat_index == 0 else 1
                block_name = f'blocks.{block_index}'
                x = InvertedResidual(
                    out_channels,
                    block_stride,
                    expand_ratio,
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
        x = nn.relu6(x)  # (batch, 7, 7, 1280)
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, 1280) -> (batch, 1280)
        logits = nn.Dense(self.num_classes, name='classifier')(x)  # (batch, 1280) -> (batch, num_classes)
        return logits


# %% [notebook-only]
# Create and run a sample ImageNet-size batch: (2, 224, 224, 3) -> (2, 1000).
example_model = MobileNetV2(num_classes=1000)
example_test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
example_variables = example_model.init(jax.random.PRNGKey(1), example_test_input, train=False)
example_logits = example_model.apply(example_variables, example_test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = MobileNetV2(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
variables = model.init(jax.random.PRNGKey(2), train_images, train=False)
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
final_loss = loss
