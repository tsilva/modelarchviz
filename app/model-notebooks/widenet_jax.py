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


# %% [notebook-only]
# Create and run one widened residual block: (2, 32, 32, 16) -> (2, 16, 16, 32).
example_block = WideBasicBlock(out_channels=32, stride=2, dropout_rate=0.0)
block_input = jnp.ones((2, 32, 32, 16))  # -> (2, 32, 32, 16)
variables = example_block.init(jax.random.PRNGKey(0), block_input, train=False)
example_block_output = example_block.apply(variables, block_input, train=False)  # (2, 32, 32, 16) -> (2, 16, 16, 32)
print("block_output shape:", example_block_output.shape)

# %%
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


# %% [notebook-only]
# Create and run a sample CIFAR-size image batch: (2, 32, 32, 3) -> (2, 10).
example_model = WideNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=10)
example_test_input = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input, train=False)
example_logits = example_model.apply(example_params, example_test_input, train=False)  # (2, 32, 32, 3) -> (2, 10)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic CIFAR-size batch.
model = WideNet(depth=10, widen_factor=1, dropout_rate=0.0, num_classes=2)
train_images = jnp.zeros((2, 32, 32, 3))  # -> (2, 32, 32, 3)
train_images = train_images.at[0, 4:16, 4:16, :].set(1.0)  # (2, 32, 32, 3)
train_images = train_images.at[1, 16:28, 16:28, :].set(1.0)  # (2, 32, 32, 3)
train_targets = jnp.array([0, 1])  # -> (2)
variables = model.init(jax.random.PRNGKey(1), train_images, train=False)
params = variables['params']
batch_stats = variables['batch_stats']


def train_step(params, batch_stats, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        current_variables = {'params': current_params, 'batch_stats': batch_stats}
        logits = model.apply(current_variables, inputs, train=False)  # (batch, 32, 32, 3) -> (batch, num_classes)
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
