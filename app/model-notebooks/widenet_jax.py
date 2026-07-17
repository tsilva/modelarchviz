# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-widebasicblock-nn-module:start
class WideBasicBlock(nn.Module):
# @arch class-widebasicblock-nn-module:end
    # @arch widebasicblock.out_channels-int:start
    out_channels: int
    # @arch widebasicblock.out_channels-int:end
    # @arch widebasicblock.stride-int-n:start
    stride: int = 1
    # @arch widebasicblock.stride-int-n:end
    # @arch widebasicblock.dropout_rate-float-n:start
    dropout_rate: float = 0.0
    # @arch widebasicblock.dropout_rate-float-n:end
    # @arch widebasicblock.use_projection-bool-false:start
    use_projection: bool = False
    # @arch widebasicblock.use_projection-bool-false:end

    # @arch widebasicblock.nn-compact:start
    @nn.compact
    # @arch widebasicblock.nn-compact:end
    # @arch widebasicblock.def-__call__-self-x-train-false:start
    def __call__(self, x, train=False):
    # @arch widebasicblock.def-__call__-self-x-train-false:end
        # Preserve the shortcut path, projecting it when width or spatial size changes.
        # @arch widebasicblock.__call__.shortcut-x:start
        shortcut = x  # (batch, height, width, channels)
        # @arch widebasicblock.__call__.shortcut-x:end
        # @arch widebasicblock.__call__.if-self-use_projection:start
        if self.use_projection:
        # @arch widebasicblock.__call__.if-self-use_projection:end
            # @arch widebasicblock.__call__.shortcut-nn-conv:start
            shortcut = nn.Conv(
            # @arch widebasicblock.__call__.shortcut-nn-conv:end
                # @arch widebasicblock.__call__.self-out_channels:start
                self.out_channels,
                # @arch widebasicblock.__call__.self-out_channels:end
                # @arch widebasicblock.__call__.n-n:start
                (1, 1),
                # @arch widebasicblock.__call__.n-n:end
                # @arch widebasicblock.__call__.strides-self-stride-self-stride:start
                strides=(self.stride, self.stride),
                # @arch widebasicblock.__call__.strides-self-stride-self-stride:end
                # @arch widebasicblock.__call__.use_bias-false:start
                use_bias=False,
                # @arch widebasicblock.__call__.use_bias-false:end
                # @arch widebasicblock.__call__.name-shortcut:start
                name='shortcut',
                # @arch widebasicblock.__call__.name-shortcut:end
            # @arch widebasicblock.__call__.x:start
            )(x)  # (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)
            # @arch widebasicblock.__call__.x:end

        # Run the widened pre-activation residual branch.
        # @arch widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-x:start
        y = nn.BatchNorm(use_running_average=not train, name='bn1')(x)  # (batch, height, width, channels)
        # @arch widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-x:end
        # @arch widebasicblock.__call__.y-nn-relu-y:start
        y = nn.relu(y)  # (batch, height, width, channels)
        # @arch widebasicblock.__call__.y-nn-relu-y:end
        # @arch widebasicblock.__call__.y-nn-conv:start
        y = nn.Conv(
        # @arch widebasicblock.__call__.y-nn-conv:end
            # @arch widebasicblock.__call__.self-out_channels.2:start
            self.out_channels,
            # @arch widebasicblock.__call__.self-out_channels.2:end
            # @arch widebasicblock.__call__.n-n.2:start
            (3, 3),
            # @arch widebasicblock.__call__.n-n.2:end
            # @arch widebasicblock.__call__.strides-self-stride-self-stride.2:start
            strides=(self.stride, self.stride),
            # @arch widebasicblock.__call__.strides-self-stride-self-stride.2:end
            # @arch widebasicblock.__call__.padding-same:start
            padding='SAME',
            # @arch widebasicblock.__call__.padding-same:end
            # @arch widebasicblock.__call__.use_bias-false.2:start
            use_bias=False,
            # @arch widebasicblock.__call__.use_bias-false.2:end
            # @arch widebasicblock.__call__.name-convn:start
            name='conv1',
            # @arch widebasicblock.__call__.name-convn:end
        # @arch widebasicblock.__call__.y:start
        )(y)  # (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)
        # @arch widebasicblock.__call__.y:end
        # @arch widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y:start
        y = nn.BatchNorm(use_running_average=not train, name='bn2')(y)  # (batch, out_h, out_w, out_channels)
        # @arch widebasicblock.__call__.y-nn-batchnorm-use_running_average-not-train-name-bnn-y:end
        # @arch widebasicblock.__call__.y-nn-relu-y.2:start
        y = nn.relu(y)  # (batch, out_h, out_w, out_channels)
        # @arch widebasicblock.__call__.y-nn-relu-y.2:end
        # @arch widebasicblock.__call__.if-self-dropout_rate-n:start
        if self.dropout_rate > 0:
        # @arch widebasicblock.__call__.if-self-dropout_rate-n:end
            # @arch widebasicblock.__call__.y-nn-dropout:start
            y = nn.Dropout(
            # @arch widebasicblock.__call__.y-nn-dropout:end
                # @arch widebasicblock.__call__.rate-self-dropout_rate:start
                rate=self.dropout_rate,
                # @arch widebasicblock.__call__.rate-self-dropout_rate:end
                # @arch widebasicblock.__call__.name-dropout:start
                name='dropout',
                # @arch widebasicblock.__call__.name-dropout:end
            # @arch widebasicblock.__call__.y-deterministic-not-train:start
            )(y, deterministic=not train)  # (batch, out_h, out_w, out_channels)
            # @arch widebasicblock.__call__.y-deterministic-not-train:end
        # @arch widebasicblock.__call__.y-nn-conv.2:start
        y = nn.Conv(
        # @arch widebasicblock.__call__.y-nn-conv.2:end
            # @arch widebasicblock.__call__.self-out_channels.3:start
            self.out_channels,
            # @arch widebasicblock.__call__.self-out_channels.3:end
            # @arch widebasicblock.__call__.n-n.3:start
            (3, 3),
            # @arch widebasicblock.__call__.n-n.3:end
            # @arch widebasicblock.__call__.padding-same.2:start
            padding='SAME',
            # @arch widebasicblock.__call__.padding-same.2:end
            # @arch widebasicblock.__call__.use_bias-false.3:start
            use_bias=False,
            # @arch widebasicblock.__call__.use_bias-false.3:end
            # @arch widebasicblock.__call__.name-convn.2:start
            name='conv2',
            # @arch widebasicblock.__call__.name-convn.2:end
        # @arch widebasicblock.__call__.y.2:start
        )(y)  # (batch, out_h, out_w, out_channels)
        # @arch widebasicblock.__call__.y.2:end

        # Merge shortcut and residual features.
        # @arch widebasicblock.__call__.y-y-shortcut:start
        y = y + shortcut  # (batch, out_h, out_w, out_channels)
        # @arch widebasicblock.__call__.y-y-shortcut:end
        # @arch widebasicblock.__call__.return-y:start
        return y
        # @arch widebasicblock.__call__.return-y:end


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
        # @arch widenet.__call__.x-nn-conv:start
        x = nn.Conv(
        # @arch widenet.__call__.x-nn-conv:end
            # @arch widenet.__call__.widths-n:start
            widths[0],
            # @arch widenet.__call__.widths-n:end
            # @arch widenet.__call__.n-n:start
            (3, 3),
            # @arch widenet.__call__.n-n:end
            # @arch widenet.__call__.padding-same:start
            padding='SAME',
            # @arch widenet.__call__.padding-same:end
            # @arch widenet.__call__.use_bias-false:start
            use_bias=False,
            # @arch widenet.__call__.use_bias-false:end
            # @arch widenet.__call__.name-convn:start
            name='conv1',
            # @arch widenet.__call__.name-convn:end
        # @arch widenet.__call__.x:start
        )(x)  # (batch, 32, 32, 3) -> (batch, 32, 32, 16)
        # @arch widenet.__call__.x:end

        # Run widened residual stages: 160, 320, then 640 channels for WRN-28-10.
        # @arch widenet.__call__.x-self-_stage:start
        x = self._stage(
        # @arch widenet.__call__.x-self-_stage:end
            # @arch widenet.__call__.x.2:start
            x,
            # @arch widenet.__call__.x.2:end
            # @arch widenet.__call__.widths-n.2:start
            widths[1],
            # @arch widenet.__call__.widths-n.2:end
            # @arch widenet.__call__.block_count:start
            block_count,
            # @arch widenet.__call__.block_count:end
            # @arch widenet.__call__.stride-n:start
            stride=1,
            # @arch widenet.__call__.stride-n:end
            # @arch widenet.__call__.train-train:start
            train=train,
            # @arch widenet.__call__.train-train:end
            # @arch widenet.__call__.name-layern:start
            name='layer1',
            # @arch widenet.__call__.name-layern:end
        # @arch widenet.__call__.code.7:start
        )  # (batch, 32, 32, 16) -> (batch, 32, 32, 160)
        # @arch widenet.__call__.code.7:end
        # @arch widenet.__call__.x-self-_stage.2:start
        x = self._stage(
        # @arch widenet.__call__.x-self-_stage.2:end
            # @arch widenet.__call__.x.3:start
            x,
            # @arch widenet.__call__.x.3:end
            # @arch widenet.__call__.widths-n.3:start
            widths[2],
            # @arch widenet.__call__.widths-n.3:end
            # @arch widenet.__call__.block_count.2:start
            block_count,
            # @arch widenet.__call__.block_count.2:end
            # @arch widenet.__call__.stride-n.2:start
            stride=2,
            # @arch widenet.__call__.stride-n.2:end
            # @arch widenet.__call__.train-train.2:start
            train=train,
            # @arch widenet.__call__.train-train.2:end
            # @arch widenet.__call__.name-layern.2:start
            name='layer2',
            # @arch widenet.__call__.name-layern.2:end
        # @arch widenet.__call__.code.8:start
        )  # (batch, 32, 32, 160) -> (batch, 16, 16, 320)
        # @arch widenet.__call__.code.8:end
        # @arch widenet.__call__.x-self-_stage.3:start
        x = self._stage(
        # @arch widenet.__call__.x-self-_stage.3:end
            # @arch widenet.__call__.x.4:start
            x,
            # @arch widenet.__call__.x.4:end
            # @arch widenet.__call__.widths-n.4:start
            widths[3],
            # @arch widenet.__call__.widths-n.4:end
            # @arch widenet.__call__.block_count.3:start
            block_count,
            # @arch widenet.__call__.block_count.3:end
            # @arch widenet.__call__.stride-n.3:start
            stride=2,
            # @arch widenet.__call__.stride-n.3:end
            # @arch widenet.__call__.train-train.3:start
            train=train,
            # @arch widenet.__call__.train-train.3:end
            # @arch widenet.__call__.name-layern.3:start
            name='layer3',
            # @arch widenet.__call__.name-layern.3:end
        # @arch widenet.__call__.code.9:start
        )  # (batch, 16, 16, 320) -> (batch, 8, 8, 640)
        # @arch widenet.__call__.code.9:end

        # Pool final feature maps and classify: (batch, 8, 8, 640) -> (batch, 10).
        # @arch widenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-bn-x:start
        x = nn.BatchNorm(use_running_average=not train, name='bn')(x)  # (batch, 8, 8, 640)
        # @arch widenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-bn-x:end
        # @arch widenet.__call__.x-nn-relu-x:start
        x = nn.relu(x)  # (batch, 8, 8, 640)
        # @arch widenet.__call__.x-nn-relu-x:end
        # @arch widenet.__call__.x-jnp-mean-x-axis-n-n:start
        x = jnp.mean(x, axis=(1, 2))  # (batch, 8, 8, 640) -> (batch, 640)
        # @arch widenet.__call__.x-jnp-mean-x-axis-n-n:end
        # @arch widenet.__call__.logits-nn-dense-self-num_classes-name-fc-x:start
        logits = nn.Dense(self.num_classes, name='fc')(x)  # (batch, 640) -> (batch, num_classes)
        # @arch widenet.__call__.logits-nn-dense-self-num_classes-name-fc-x:end
        return logits

    def _stage(self, x, channels, blocks, stride, train, name):
        # Start each stage with the only block that may widen channels or downsample.
        # @arch widenet._stage.for-index-in-range-blocks:start
        for index in range(blocks):
        # @arch widenet._stage.for-index-in-range-blocks:end
            # @arch widenet._stage.block_stride-stride-if-index-n-else-n:start
            block_stride = stride if index == 0 else 1
            # @arch widenet._stage.block_stride-stride-if-index-n-else-n:end
            # @arch widenet._stage.use_projection-index-n:start
            use_projection = index == 0
            # @arch widenet._stage.use_projection-index-n:end
            # @arch widenet._stage.block_name-f-name-index:start
            block_name = f'{name}.{index}'
            # @arch widenet._stage.block_name-f-name-index:end
            # @arch widenet._stage.x-widebasicblock:start
            x = WideBasicBlock(
            # @arch widenet._stage.x-widebasicblock:end
                # @arch widenet._stage.channels:start
                channels,
                # @arch widenet._stage.channels:end
                # @arch widenet._stage.stride-block_stride:start
                stride=block_stride,
                # @arch widenet._stage.stride-block_stride:end
                # @arch widenet._stage.dropout_rate-self-dropout_rate:start
                dropout_rate=self.dropout_rate,
                # @arch widenet._stage.dropout_rate-self-dropout_rate:end
                # @arch widenet._stage.use_projection-use_projection:start
                use_projection=use_projection,
                # @arch widenet._stage.use_projection-use_projection:end
                # @arch widenet._stage.name-block_name:start
                name=block_name,
                # @arch widenet._stage.name-block_name:end
            # @arch widenet._stage.x-train-train:start
            )(x, train=train)  # (batch, height, width, channels) -> (batch, out_h, out_w, channels)
            # @arch widenet._stage.x-train-train:end
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
