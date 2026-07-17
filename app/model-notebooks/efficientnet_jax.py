# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-squeezeexcite-nn-module:start
class SqueezeExcite(nn.Module):
# @arch class-squeezeexcite-nn-module:end
    # @arch squeezeexcite.squeeze_channels-int:start
    squeeze_channels: int
    # @arch squeezeexcite.squeeze_channels-int:end

    # @arch squeezeexcite.nn-compact:start
    @nn.compact
    # @arch squeezeexcite.nn-compact:end
    # @arch squeezeexcite.def-__call__-self-x:start
    def __call__(self, x):
    # @arch squeezeexcite.def-__call__-self-x:end
        # Squeeze spatial dimensions into one descriptor per channel.
        # @arch squeezeexcite.__call__.scale-jnp-mean-x-axis-n-n-keepdims-true:start
        scale = jnp.mean(x, axis=(1, 2), keepdims=True)  # (batch, height, width, channels) -> (batch, 1, 1, channels)
        # @arch squeezeexcite.__call__.scale-jnp-mean-x-axis-n-n-keepdims-true:end

        # Excite channels and gate the original feature map.
        # @arch squeezeexcite.__call__.scale-nn-conv-self-squeeze_channels-n-n-name-reduce-scale:start
        scale = nn.Conv(self.squeeze_channels, (1, 1), name='reduce')(scale)  # (batch, 1, 1, channels) -> (batch, 1, 1, squeeze_channels)
        # @arch squeezeexcite.__call__.scale-nn-conv-self-squeeze_channels-n-n-name-reduce-scale:end
        # @arch squeezeexcite.__call__.scale-nn-silu-scale:start
        scale = nn.silu(scale)  # (batch, 1, 1, squeeze_channels)
        # @arch squeezeexcite.__call__.scale-nn-silu-scale:end
        # @arch squeezeexcite.__call__.channel_count-x-shape-n:start
        channel_count = x.shape[-1]  # (batch, height, width, channels) -> scalar
        # @arch squeezeexcite.__call__.channel_count-x-shape-n:end
        # @arch squeezeexcite.__call__.scale-nn-conv-channel_count-n-n-name-expand-scale:start
        scale = nn.Conv(channel_count, (1, 1), name='expand')(scale)  # (batch, 1, 1, squeeze_channels) -> (batch, 1, 1, channels)
        # @arch squeezeexcite.__call__.scale-nn-conv-channel_count-n-n-name-expand-scale:end
        # @arch squeezeexcite.__call__.scale-nn-sigmoid-scale:start
        scale = nn.sigmoid(scale)  # (batch, 1, 1, channels)
        # @arch squeezeexcite.__call__.scale-nn-sigmoid-scale:end
        # @arch squeezeexcite.__call__.y-x-scale:start
        y = x * scale  # (batch, height, width, channels)
        # @arch squeezeexcite.__call__.y-x-scale:end
        # @arch squeezeexcite.__call__.return-y:start
        return y
        # @arch squeezeexcite.__call__.return-y:end


# %% [notebook-only]
# Create and run a squeeze-excitation gate: (2, 16, 16, 8) -> (2, 16, 16, 8).
gate = SqueezeExcite(squeeze_channels=2)
feature_map = jnp.ones((2, 16, 16, 8))  # -> (2, 16, 16, 8)
example_params = gate.init(jax.random.PRNGKey(0), feature_map)
gated = gate.apply(example_params, feature_map)  # (2, 16, 16, 8) -> (2, 16, 16, 8)
print("gated shape:", gated.shape)

# %%
# @arch class-mbconv-nn-module:start
class MBConv(nn.Module):
# @arch class-mbconv-nn-module:end
    # @arch mbconv.out_channels-int:start
    out_channels: int
    # @arch mbconv.out_channels-int:end
    # @arch mbconv.expand_ratio-int:start
    expand_ratio: int
    # @arch mbconv.expand_ratio-int:end
    # @arch mbconv.stride-int:start
    stride: int
    # @arch mbconv.stride-int:end
    # @arch mbconv.kernel_size-int:start
    kernel_size: int
    # @arch mbconv.kernel_size-int:end
    # @arch mbconv.se_ratio-float-n:start
    se_ratio: float = 0.25
    # @arch mbconv.se_ratio-float-n:end

    # @arch mbconv.nn-compact:start
    @nn.compact
    # @arch mbconv.nn-compact:end
    # @arch mbconv.def-__call__-self-x-train-false:start
    def __call__(self, x, train=False):
    # @arch mbconv.def-__call__-self-x-train-false:end
        # Configure expanded and squeezed channel widths.
        # @arch mbconv.__call__.in_channels-x-shape-n:start
        in_channels = x.shape[-1]  # (batch, height, width, in_channels) -> scalar
        # @arch mbconv.__call__.in_channels-x-shape-n:end
        # @arch mbconv.__call__.expanded_channels-in_channels-self-expand_ratio:start
        expanded_channels = in_channels * self.expand_ratio
        # @arch mbconv.__call__.expanded_channels-in_channels-self-expand_ratio:end
        # @arch mbconv.__call__.squeeze_channels-max-n-int-expanded_channels-self-se_ratio:start
        squeeze_channels = max(1, int(expanded_channels * self.se_ratio))
        # @arch mbconv.__call__.squeeze_channels-max-n-int-expanded_channels-self-se_ratio:end
        # @arch mbconv.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels:start
        use_residual = self.stride == 1 and in_channels == self.out_channels
        # @arch mbconv.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels:end

        # Expand channels before depthwise spatial filtering.
        # @arch mbconv.__call__.y-x:start
        y = x  # (batch, height, width, in_channels)
        # @arch mbconv.__call__.y-x:end
        # @arch mbconv.__call__.if-self-expand_ratio-n:start
        if self.expand_ratio != 1:
        # @arch mbconv.__call__.if-self-expand_ratio-n:end
            # @arch mbconv.__call__.y-nn-conv:start
            y = nn.Conv(
            # @arch mbconv.__call__.y-nn-conv:end
                # @arch mbconv.__call__.expanded_channels:start
                expanded_channels,
                # @arch mbconv.__call__.expanded_channels:end
                # @arch mbconv.__call__.n-n:start
                (1, 1),
                # @arch mbconv.__call__.n-n:end
                # @arch mbconv.__call__.use_bias-false:start
                use_bias=False,
                # @arch mbconv.__call__.use_bias-false:end
                # @arch mbconv.__call__.name-expand_conv:start
                name='expand_conv',
                # @arch mbconv.__call__.name-expand_conv:end
            # @arch mbconv.__call__.y:start
            )(y)  # (batch, height, width, in_channels) -> (batch, height, width, expanded_channels)
            # @arch mbconv.__call__.y:end
            # @arch mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y:start
            y = nn.BatchNorm(use_running_average=not train, name='expand_bn')(y)  # (batch, height, width, expanded_channels)
            # @arch mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y:end
            # @arch mbconv.__call__.y-nn-silu-y:start
            y = nn.silu(y)  # (batch, height, width, expanded_channels)
            # @arch mbconv.__call__.y-nn-silu-y:end

        # Apply one depthwise filter per channel.
        # @arch mbconv.__call__.y-nn-conv.2:start
        y = nn.Conv(
        # @arch mbconv.__call__.y-nn-conv.2:end
            # @arch mbconv.__call__.expanded_channels.2:start
            expanded_channels,
            # @arch mbconv.__call__.expanded_channels.2:end
            # @arch mbconv.__call__.self-kernel_size-self-kernel_size:start
            (self.kernel_size, self.kernel_size),
            # @arch mbconv.__call__.self-kernel_size-self-kernel_size:end
            # @arch mbconv.__call__.strides-self-stride-self-stride:start
            strides=(self.stride, self.stride),
            # @arch mbconv.__call__.strides-self-stride-self-stride:end
            # @arch mbconv.__call__.padding-same:start
            padding='SAME',
            # @arch mbconv.__call__.padding-same:end
            # @arch mbconv.__call__.feature_group_count-expanded_channels:start
            feature_group_count=expanded_channels,
            # @arch mbconv.__call__.feature_group_count-expanded_channels:end
            # @arch mbconv.__call__.use_bias-false.2:start
            use_bias=False,
            # @arch mbconv.__call__.use_bias-false.2:end
            # @arch mbconv.__call__.name-depthwise_conv:start
            name='depthwise_conv',
            # @arch mbconv.__call__.name-depthwise_conv:end
        # @arch mbconv.__call__.y.2:start
        )(y)  # (batch, height, width, expanded_channels) -> (batch, out_h, out_w, expanded_channels)
        # @arch mbconv.__call__.y.2:end
        # @arch mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y:start
        y = nn.BatchNorm(use_running_average=not train, name='depthwise_bn')(y)  # (batch, out_h, out_w, expanded_channels)
        # @arch mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y:end
        # @arch mbconv.__call__.y-nn-silu-y.2:start
        y = nn.silu(y)  # (batch, out_h, out_w, expanded_channels)
        # @arch mbconv.__call__.y-nn-silu-y.2:end

        # Reweight channels with squeeze-and-excitation, then project back.
        # @arch mbconv.__call__.y-squeezeexcite-squeeze_channels-name-se-y:start
        y = SqueezeExcite(squeeze_channels, name='se')(y)  # (batch, out_h, out_w, expanded_channels)
        # @arch mbconv.__call__.y-squeezeexcite-squeeze_channels-name-se-y:end
        # @arch mbconv.__call__.y-nn-conv.3:start
        y = nn.Conv(
        # @arch mbconv.__call__.y-nn-conv.3:end
            # @arch mbconv.__call__.self-out_channels:start
            self.out_channels,
            # @arch mbconv.__call__.self-out_channels:end
            # @arch mbconv.__call__.n-n.2:start
            (1, 1),
            # @arch mbconv.__call__.n-n.2:end
            # @arch mbconv.__call__.use_bias-false.3:start
            use_bias=False,
            # @arch mbconv.__call__.use_bias-false.3:end
            # @arch mbconv.__call__.name-project_conv:start
            name='project_conv',
            # @arch mbconv.__call__.name-project_conv:end
        # @arch mbconv.__call__.y.3:start
        )(y)  # (batch, out_h, out_w, expanded_channels) -> (batch, out_h, out_w, out_channels)
        # @arch mbconv.__call__.y.3:end
        # @arch mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y:start
        y = nn.BatchNorm(use_running_average=not train, name='project_bn')(y)  # (batch, out_h, out_w, out_channels)
        # @arch mbconv.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y:end

        # Add the residual shortcut only for same-shape blocks.
        # @arch mbconv.__call__.if-use_residual:start
        if use_residual:
        # @arch mbconv.__call__.if-use_residual:end
            # @arch mbconv.__call__.y-y-x:start
            y = y + x  # (batch, height, width, out_channels)
            # @arch mbconv.__call__.y-y-x:end
        # @arch mbconv.__call__.return-y:start
        return y
        # @arch mbconv.__call__.return-y:end


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
            # @arch efficientnet.__call__.n-n-n-n-n:start
            (1, 16, 1, 1, 3),
            # @arch efficientnet.__call__.n-n-n-n-n:end
            # @arch efficientnet.__call__.n-n-n-n-n.2:start
            (6, 24, 2, 2, 3),
            # @arch efficientnet.__call__.n-n-n-n-n.2:end
            # @arch efficientnet.__call__.n-n-n-n-n.3:start
            (6, 40, 2, 2, 5),
            # @arch efficientnet.__call__.n-n-n-n-n.3:end
            # @arch efficientnet.__call__.n-n-n-n-n.4:start
            (6, 80, 3, 2, 3),
            # @arch efficientnet.__call__.n-n-n-n-n.4:end
            # @arch efficientnet.__call__.n-n-n-n-n.5:start
            (6, 112, 3, 1, 5),
            # @arch efficientnet.__call__.n-n-n-n-n.5:end
            # @arch efficientnet.__call__.n-n-n-n-n.6:start
            (6, 192, 4, 2, 5),
            # @arch efficientnet.__call__.n-n-n-n-n.6:end
            # @arch efficientnet.__call__.n-n-n-n-n.7:start
            (6, 320, 1, 1, 3),
            # @arch efficientnet.__call__.n-n-n-n-n.7:end
        ]

        # Convert image input into stem features: (batch, 224, 224, 3) -> (batch, 112, 112, 32).
        # @arch efficientnet.__call__.x-nn-conv:start
        x = nn.Conv(
        # @arch efficientnet.__call__.x-nn-conv:end
            # @arch efficientnet.__call__.n:start
            32,
            # @arch efficientnet.__call__.n:end
            # @arch efficientnet.__call__.n-n:start
            (3, 3),
            # @arch efficientnet.__call__.n-n:end
            # @arch efficientnet.__call__.strides-n-n:start
            strides=(2, 2),
            # @arch efficientnet.__call__.strides-n-n:end
            # @arch efficientnet.__call__.padding-same:start
            padding='SAME',
            # @arch efficientnet.__call__.padding-same:end
            # @arch efficientnet.__call__.use_bias-false:start
            use_bias=False,
            # @arch efficientnet.__call__.use_bias-false:end
            # @arch efficientnet.__call__.name-stem_conv:start
            name='stem_conv',
            # @arch efficientnet.__call__.name-stem_conv:end
        # @arch efficientnet.__call__.x:start
        )(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, 32)
        # @arch efficientnet.__call__.x:end
        # @arch efficientnet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x:start
        x = nn.BatchNorm(use_running_average=not train, name='stem_bn')(x)  # (batch, 112, 112, 32)
        # @arch efficientnet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x:end
        # @arch efficientnet.__call__.x-nn-silu-x:start
        x = nn.silu(x)  # (batch, 112, 112, 32)
        # @arch efficientnet.__call__.x-nn-silu-x:end

        # Run compound-scaled MBConv stages with depthwise filters and SE gates.
        block_index = 0
        # @arch efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings:start
        for expand_ratio, out_channels, repeats, stride, kernel_size in settings:
        # @arch efficientnet.__call__.for-expand_ratio-out_channels-repeats-stride-kernel_size-in-settings:end
            # @arch efficientnet.__call__.for-repeat_index-in-range-repeats:start
            for repeat_index in range(repeats):
            # @arch efficientnet.__call__.for-repeat_index-in-range-repeats:end
                # @arch efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n:start
                block_stride = stride if repeat_index == 0 else 1
                # @arch efficientnet.__call__.block_stride-stride-if-repeat_index-n-else-n:end
                # @arch efficientnet.__call__.block_name-f-blocks-block_index:start
                block_name = f'blocks.{block_index}'
                # @arch efficientnet.__call__.block_name-f-blocks-block_index:end
                # @arch efficientnet.__call__.x-mbconv:start
                x = MBConv(
                # @arch efficientnet.__call__.x-mbconv:end
                    # @arch efficientnet.__call__.out_channels:start
                    out_channels,
                    # @arch efficientnet.__call__.out_channels:end
                    # @arch efficientnet.__call__.expand_ratio:start
                    expand_ratio,
                    # @arch efficientnet.__call__.expand_ratio:end
                    # @arch efficientnet.__call__.block_stride:start
                    block_stride,
                    # @arch efficientnet.__call__.block_stride:end
                    # @arch efficientnet.__call__.kernel_size:start
                    kernel_size,
                    # @arch efficientnet.__call__.kernel_size:end
                    # @arch efficientnet.__call__.name-block_name:start
                    name=block_name,
                    # @arch efficientnet.__call__.name-block_name:end
                # @arch efficientnet.__call__.x-train-train:start
                )(x, train=train)  # (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)
                # @arch efficientnet.__call__.x-train-train:end
                # @arch efficientnet.__call__.block_index-block_index-n:start
                block_index = block_index + 1
                # @arch efficientnet.__call__.block_index-block_index-n:end

        # Expand final channels, pool, and classify.
        # @arch efficientnet.__call__.x-nn-conv.2:start
        x = nn.Conv(
        # @arch efficientnet.__call__.x-nn-conv.2:end
            # @arch efficientnet.__call__.n.2:start
            1280,
            # @arch efficientnet.__call__.n.2:end
            # @arch efficientnet.__call__.n-n.2:start
            (1, 1),
            # @arch efficientnet.__call__.n-n.2:end
            # @arch efficientnet.__call__.use_bias-false.2:start
            use_bias=False,
            # @arch efficientnet.__call__.use_bias-false.2:end
            # @arch efficientnet.__call__.name-head_conv:start
            name='head_conv',
            # @arch efficientnet.__call__.name-head_conv:end
        # @arch efficientnet.__call__.x.2:start
        )(x)  # (batch, 7, 7, 320) -> (batch, 7, 7, 1280)
        # @arch efficientnet.__call__.x.2:end
        # @arch efficientnet.__call__.x-nn-batchnorm-use_running_average-not-train-name-head_bn-x:start
        x = nn.BatchNorm(use_running_average=not train, name='head_bn')(x)  # (batch, 7, 7, 1280)
        # @arch efficientnet.__call__.x-nn-batchnorm-use_running_average-not-train-name-head_bn-x:end
        # @arch efficientnet.__call__.x-nn-silu-x.2:start
        x = nn.silu(x)  # (batch, 7, 7, 1280)
        # @arch efficientnet.__call__.x-nn-silu-x.2:end
        # @arch efficientnet.__call__.x-jnp-mean-x-axis-n-n:start
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, 1280) -> (batch, 1280)
        # @arch efficientnet.__call__.x-jnp-mean-x-axis-n-n:end
        # @arch efficientnet.__call__.logits-nn-dense-self-num_classes-name-classifier-x:start
        logits = nn.Dense(self.num_classes, name='classifier')(x)  # (batch, 1280) -> (batch, num_classes)
        # @arch efficientnet.__call__.logits-nn-dense-self-num_classes-name-classifier-x:end
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
