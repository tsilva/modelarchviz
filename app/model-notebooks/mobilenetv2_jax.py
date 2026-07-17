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
        # @arch invertedresidual.__call__.in_channels-x-shape-n:start
        in_channels = x.shape[-1]  # (batch, height, width, in_channels) -> scalar
        # @arch invertedresidual.__call__.in_channels-x-shape-n:end
        # @arch invertedresidual.__call__.hidden_channels-in_channels-self-expand_ratio:start
        hidden_channels = in_channels * self.expand_ratio
        # @arch invertedresidual.__call__.hidden_channels-in_channels-self-expand_ratio:end
        # @arch invertedresidual.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels:start
        use_residual = self.stride == 1 and in_channels == self.out_channels
        # @arch invertedresidual.__call__.use_residual-self-stride-n-and-in_channels-self-out_channels:end

        # Expand narrow input bottlenecks when the expansion ratio is greater than one.
        # @arch invertedresidual.__call__.y-x:start
        y = x  # (batch, height, width, in_channels)
        # @arch invertedresidual.__call__.y-x:end
        # @arch invertedresidual.__call__.if-self-expand_ratio-n:start
        if self.expand_ratio != 1:
        # @arch invertedresidual.__call__.if-self-expand_ratio-n:end
            # @arch invertedresidual.__call__.y-nn-conv:start
            y = nn.Conv(
            # @arch invertedresidual.__call__.y-nn-conv:end
                # @arch invertedresidual.__call__.hidden_channels:start
                hidden_channels,
                # @arch invertedresidual.__call__.hidden_channels:end
                # @arch invertedresidual.__call__.n-n:start
                (1, 1),
                # @arch invertedresidual.__call__.n-n:end
                # @arch invertedresidual.__call__.use_bias-false:start
                use_bias=False,
                # @arch invertedresidual.__call__.use_bias-false:end
                # @arch invertedresidual.__call__.name-expand_conv:start
                name='expand_conv',
                # @arch invertedresidual.__call__.name-expand_conv:end
            # @arch invertedresidual.__call__.y:start
            )(y)  # (batch, height, width, in_channels) -> (batch, height, width, hidden_channels)
            # @arch invertedresidual.__call__.y:end
            # @arch invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y:start
            y = nn.BatchNorm(use_running_average=not train, name='expand_bn')(y)  # (batch, height, width, hidden_channels)
            # @arch invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-expand_bn-y:end
            # @arch invertedresidual.__call__.y-nn-relun-y:start
            y = nn.relu6(y)  # (batch, height, width, hidden_channels)
            # @arch invertedresidual.__call__.y-nn-relun-y:end

        # Apply one spatial 3x3 filter per expanded channel.
        # @arch invertedresidual.__call__.y-nn-conv.2:start
        y = nn.Conv(
        # @arch invertedresidual.__call__.y-nn-conv.2:end
            # @arch invertedresidual.__call__.hidden_channels.2:start
            hidden_channels,
            # @arch invertedresidual.__call__.hidden_channels.2:end
            # @arch invertedresidual.__call__.n-n.2:start
            (3, 3),
            # @arch invertedresidual.__call__.n-n.2:end
            # @arch invertedresidual.__call__.strides-self-stride-self-stride:start
            strides=(self.stride, self.stride),
            # @arch invertedresidual.__call__.strides-self-stride-self-stride:end
            # @arch invertedresidual.__call__.padding-same:start
            padding='SAME',
            # @arch invertedresidual.__call__.padding-same:end
            # @arch invertedresidual.__call__.feature_group_count-hidden_channels:start
            feature_group_count=hidden_channels,
            # @arch invertedresidual.__call__.feature_group_count-hidden_channels:end
            # @arch invertedresidual.__call__.use_bias-false.2:start
            use_bias=False,
            # @arch invertedresidual.__call__.use_bias-false.2:end
            # @arch invertedresidual.__call__.name-depthwise_conv:start
            name='depthwise_conv',
            # @arch invertedresidual.__call__.name-depthwise_conv:end
        # @arch invertedresidual.__call__.y.2:start
        )(y)  # (batch, height, width, hidden_channels) -> (batch, out_h, out_w, hidden_channels)
        # @arch invertedresidual.__call__.y.2:end
        # @arch invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y:start
        y = nn.BatchNorm(use_running_average=not train, name='depthwise_bn')(y)  # (batch, out_h, out_w, hidden_channels)
        # @arch invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-depthwise_bn-y:end
        # @arch invertedresidual.__call__.y-nn-relun-y.2:start
        y = nn.relu6(y)  # (batch, out_h, out_w, hidden_channels)
        # @arch invertedresidual.__call__.y-nn-relun-y.2:end

        # Project back to a narrow linear bottleneck without an activation.
        # @arch invertedresidual.__call__.y-nn-conv.3:start
        y = nn.Conv(
        # @arch invertedresidual.__call__.y-nn-conv.3:end
            # @arch invertedresidual.__call__.self-out_channels:start
            self.out_channels,
            # @arch invertedresidual.__call__.self-out_channels:end
            # @arch invertedresidual.__call__.n-n.3:start
            (1, 1),
            # @arch invertedresidual.__call__.n-n.3:end
            # @arch invertedresidual.__call__.use_bias-false.3:start
            use_bias=False,
            # @arch invertedresidual.__call__.use_bias-false.3:end
            # @arch invertedresidual.__call__.name-project_conv:start
            name='project_conv',
            # @arch invertedresidual.__call__.name-project_conv:end
        # @arch invertedresidual.__call__.y.3:start
        )(y)  # (batch, out_h, out_w, hidden_channels) -> (batch, out_h, out_w, out_channels)
        # @arch invertedresidual.__call__.y.3:end
        # @arch invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y:start
        y = nn.BatchNorm(use_running_average=not train, name='project_bn')(y)  # (batch, out_h, out_w, out_channels)
        # @arch invertedresidual.__call__.y-nn-batchnorm-use_running_average-not-train-name-project_bn-y:end

        # Add the shortcut only when bottleneck shapes match.
        # @arch invertedresidual.__call__.if-use_residual:start
        if use_residual:
        # @arch invertedresidual.__call__.if-use_residual:end
            # @arch invertedresidual.__call__.y-y-x:start
            y = y + x  # (batch, height, width, out_channels)
            # @arch invertedresidual.__call__.y-y-x:end
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
    # @arch mobilenetvn.dropout_rate-float-n:start
    dropout_rate: float = 0.2
    # @arch mobilenetvn.dropout_rate-float-n:end

    @nn.compact
    def __call__(self, x, train=False):
        # MobileNetV2 stage plan: expansion, output channels, repeats, first stride.
        settings = [
            # @arch mobilenetvn.__call__.n-n-n-n:start
            (1, 16, 1, 1),
            # @arch mobilenetvn.__call__.n-n-n-n:end
            # @arch mobilenetvn.__call__.n-n-n-n.2:start
            (6, 24, 2, 2),
            # @arch mobilenetvn.__call__.n-n-n-n.2:end
            # @arch mobilenetvn.__call__.n-n-n-n.3:start
            (6, 32, 3, 2),
            # @arch mobilenetvn.__call__.n-n-n-n.3:end
            # @arch mobilenetvn.__call__.n-n-n-n.4:start
            (6, 64, 4, 2),
            # @arch mobilenetvn.__call__.n-n-n-n.4:end
            # @arch mobilenetvn.__call__.n-n-n-n.5:start
            (6, 96, 3, 1),
            # @arch mobilenetvn.__call__.n-n-n-n.5:end
            # @arch mobilenetvn.__call__.n-n-n-n.6:start
            (6, 160, 3, 2),
            # @arch mobilenetvn.__call__.n-n-n-n.6:end
            # @arch mobilenetvn.__call__.n-n-n-n.7:start
            (6, 320, 1, 1),
            # @arch mobilenetvn.__call__.n-n-n-n.7:end
        ]

        # Convert image input into stem features: (batch, 224, 224, 3) -> (batch, 112, 112, 32).
        # @arch mobilenetvn.__call__.x-nn-conv:start
        x = nn.Conv(
        # @arch mobilenetvn.__call__.x-nn-conv:end
            # @arch mobilenetvn.__call__.n:start
            32,
            # @arch mobilenetvn.__call__.n:end
            # @arch mobilenetvn.__call__.n-n:start
            (3, 3),
            # @arch mobilenetvn.__call__.n-n:end
            # @arch mobilenetvn.__call__.strides-n-n:start
            strides=(2, 2),
            # @arch mobilenetvn.__call__.strides-n-n:end
            # @arch mobilenetvn.__call__.padding-same:start
            padding='SAME',
            # @arch mobilenetvn.__call__.padding-same:end
            # @arch mobilenetvn.__call__.use_bias-false:start
            use_bias=False,
            # @arch mobilenetvn.__call__.use_bias-false:end
            # @arch mobilenetvn.__call__.name-stem_conv:start
            name='stem_conv',
            # @arch mobilenetvn.__call__.name-stem_conv:end
        # @arch mobilenetvn.__call__.x:start
        )(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, 32)
        # @arch mobilenetvn.__call__.x:end
        # @arch mobilenetvn.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x:start
        x = nn.BatchNorm(use_running_average=not train, name='stem_bn')(x)  # (batch, 112, 112, 32)
        # @arch mobilenetvn.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_bn-x:end
        # @arch mobilenetvn.__call__.x-nn-relun-x:start
        x = nn.relu6(x)  # (batch, 112, 112, 32)
        # @arch mobilenetvn.__call__.x-nn-relun-x:end

        # Run inverted residual stages with depthwise filters and linear bottlenecks.
        block_index = 0
        # @arch mobilenetvn.__call__.for-expand_ratio-out_channels-repeats-stride-in-settings:start
        for expand_ratio, out_channels, repeats, stride in settings:
        # @arch mobilenetvn.__call__.for-expand_ratio-out_channels-repeats-stride-in-settings:end
            # @arch mobilenetvn.__call__.for-repeat_index-in-range-repeats:start
            for repeat_index in range(repeats):
            # @arch mobilenetvn.__call__.for-repeat_index-in-range-repeats:end
                # @arch mobilenetvn.__call__.block_stride-stride-if-repeat_index-n-else-n:start
                block_stride = stride if repeat_index == 0 else 1
                # @arch mobilenetvn.__call__.block_stride-stride-if-repeat_index-n-else-n:end
                # @arch mobilenetvn.__call__.block_name-f-blocks-block_index:start
                block_name = f'blocks.{block_index}'
                # @arch mobilenetvn.__call__.block_name-f-blocks-block_index:end
                # @arch mobilenetvn.__call__.x-invertedresidual:start
                x = InvertedResidual(
                # @arch mobilenetvn.__call__.x-invertedresidual:end
                    # @arch mobilenetvn.__call__.out_channels:start
                    out_channels,
                    # @arch mobilenetvn.__call__.out_channels:end
                    # @arch mobilenetvn.__call__.block_stride:start
                    block_stride,
                    # @arch mobilenetvn.__call__.block_stride:end
                    # @arch mobilenetvn.__call__.expand_ratio:start
                    expand_ratio,
                    # @arch mobilenetvn.__call__.expand_ratio:end
                    # @arch mobilenetvn.__call__.name-block_name:start
                    name=block_name,
                    # @arch mobilenetvn.__call__.name-block_name:end
                # @arch mobilenetvn.__call__.x-train-train:start
                )(x, train=train)  # (batch, height, width, channels) -> (batch, out_h, out_w, out_channels)
                # @arch mobilenetvn.__call__.x-train-train:end
                block_index = block_index + 1

        # Expand final channels, pool, regularize, and classify.
        # @arch mobilenetvn.__call__.x-nn-conv.2:start
        x = nn.Conv(
        # @arch mobilenetvn.__call__.x-nn-conv.2:end
            # @arch mobilenetvn.__call__.n.2:start
            1280,
            # @arch mobilenetvn.__call__.n.2:end
            # @arch mobilenetvn.__call__.n-n.2:start
            (1, 1),
            # @arch mobilenetvn.__call__.n-n.2:end
            # @arch mobilenetvn.__call__.use_bias-false.2:start
            use_bias=False,
            # @arch mobilenetvn.__call__.use_bias-false.2:end
            # @arch mobilenetvn.__call__.name-head_conv:start
            name='head_conv',
            # @arch mobilenetvn.__call__.name-head_conv:end
        # @arch mobilenetvn.__call__.x.2:start
        )(x)  # (batch, 7, 7, 320) -> (batch, 7, 7, 1280)
        # @arch mobilenetvn.__call__.x.2:end
        # @arch mobilenetvn.__call__.x-nn-batchnorm-use_running_average-not-train-name-head_bn-x:start
        x = nn.BatchNorm(use_running_average=not train, name='head_bn')(x)  # (batch, 7, 7, 1280)
        # @arch mobilenetvn.__call__.x-nn-batchnorm-use_running_average-not-train-name-head_bn-x:end
        # @arch mobilenetvn.__call__.x-nn-relun-x.2:start
        x = nn.relu6(x)  # (batch, 7, 7, 1280)
        # @arch mobilenetvn.__call__.x-nn-relun-x.2:end
        # @arch mobilenetvn.__call__.x-jnp-mean-x-axis-n-n:start
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, 1280) -> (batch, 1280)
        # @arch mobilenetvn.__call__.x-jnp-mean-x-axis-n-n:end
        # @arch mobilenetvn.__call__.x-nn-dropout-rate-self-dropout_rate-deterministic-not-train-name-dropout:start
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train, name='dropout')(x)  # (batch, 1280)
        # @arch mobilenetvn.__call__.x-nn-dropout-rate-self-dropout_rate-deterministic-not-train-name-dropout:end
        # @arch mobilenetvn.__call__.logits-nn-dense-self-num_classes-name-classifier-x:start
        logits = nn.Dense(self.num_classes, name='classifier')(x)  # (batch, 1280) -> (batch, num_classes)
        # @arch mobilenetvn.__call__.logits-nn-dense-self-num_classes-name-classifier-x:end
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
# @arch train_images-jnp-zeros-n-n-n-n:start
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
# @arch train_images-jnp-zeros-n-n-n-n:end
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
variables = model.init(jax.random.PRNGKey(2), train_images, train=False)
params = variables['params']
batch_stats = variables['batch_stats']


def train_step(params, batch_stats, inputs, targets, dropout_key, learning_rate=0.01):
    def loss_fn(current_params):
        current_variables = {'params': current_params, 'batch_stats': batch_stats}
        # @arch train_step.loss_fn.logits-updated_variables-model-apply:start
        logits, updated_variables = model.apply(
        # @arch train_step.loss_fn.logits-updated_variables-model-apply:end
            current_variables,
            # @arch train_step.loss_fn.inputs:start
            inputs,
            # @arch train_step.loss_fn.inputs:end
            train=True,
            rngs={'dropout': dropout_key},
            mutable=['batch_stats'],
        )  # (batch, 224, 224, 3) -> (batch, num_classes)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, num_classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, num_classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, num_classes) -> scalar
        updated_batch_stats = updated_variables['batch_stats']
        return loss, updated_batch_stats

    loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, updated_batch_stats), grads = loss_and_grad_fn(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, updated_batch_stats, loss


# Fit the model for a few steps on the tiny dataset.
training_key = jax.random.PRNGKey(3)
for step in range(3):
    training_key, dropout_key = jax.random.split(training_key)
    params, batch_stats, loss = train_step(
        params,
        batch_stats,
        train_images,
        train_targets,
        dropout_key,
    )

# Keep the final scalar loss for inspection.
final_loss = loss
