# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-denselayer-nn-module:start
class DenseLayer(nn.Module):
# @arch class-denselayer-nn-module:end
    growth_rate: int
    bottleneck_width: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train=False):
        # Compress existing features, then produce new growth features.
        bottleneck_channels = self.bottleneck_width * self.growth_rate
        # @arch denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-x:start
        y = nn.BatchNorm(use_running_average=not train, name='norm1')(x)  # (batch, height, width, channels)
        # @arch denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-x:end
        # @arch denselayer.__call__.y-nn-relu-y:start
        y = nn.relu(y)  # (batch, height, width, channels)
        # @arch denselayer.__call__.y-nn-relu-y:end
        # @arch denselayer.__call__.y-nn-conv:start
        y = nn.Conv(
        # @arch denselayer.__call__.y-nn-conv:end
            # @arch denselayer.__call__.bottleneck_channels:start
            bottleneck_channels,
            # @arch denselayer.__call__.bottleneck_channels:end
            # @arch denselayer.__call__.n-n:start
            (1, 1),
            # @arch denselayer.__call__.n-n:end
            # @arch denselayer.__call__.use_bias-false:start
            use_bias=False,
            # @arch denselayer.__call__.use_bias-false:end
            # @arch denselayer.__call__.name-convn:start
            name='conv1',
            # @arch denselayer.__call__.name-convn:end
        # @arch denselayer.__call__.y:start
        )(y)  # (batch, height, width, channels) -> (batch, height, width, bottleneck_channels)
        # @arch denselayer.__call__.y:end
        # @arch denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-y:start
        y = nn.BatchNorm(use_running_average=not train, name='norm2')(y)  # (batch, height, width, bottleneck_channels)
        # @arch denselayer.__call__.y-nn-batchnorm-use_running_average-not-train-name-normn-y:end
        # @arch denselayer.__call__.y-nn-relu-y.2:start
        y = nn.relu(y)  # (batch, height, width, bottleneck_channels)
        # @arch denselayer.__call__.y-nn-relu-y.2:end
        # @arch denselayer.__call__.y-nn-conv.2:start
        y = nn.Conv(
        # @arch denselayer.__call__.y-nn-conv.2:end
            # @arch denselayer.__call__.self-growth_rate:start
            self.growth_rate,
            # @arch denselayer.__call__.self-growth_rate:end
            # @arch denselayer.__call__.n-n.2:start
            (3, 3),
            # @arch denselayer.__call__.n-n.2:end
            # @arch denselayer.__call__.padding-same:start
            padding='SAME',
            # @arch denselayer.__call__.padding-same:end
            # @arch denselayer.__call__.use_bias-false.2:start
            use_bias=False,
            # @arch denselayer.__call__.use_bias-false.2:end
            # @arch denselayer.__call__.name-convn.2:start
            name='conv2',
            # @arch denselayer.__call__.name-convn.2:end
        # @arch denselayer.__call__.y.2:start
        )(y)  # (batch, height, width, bottleneck_channels) -> (batch, height, width, growth_rate)
        # @arch denselayer.__call__.y.2:end
        if self.dropout_rate > 0:
            y = nn.Dropout(
                rate=self.dropout_rate,
                name='dropout',
            )(y, deterministic=not train)  # (batch, height, width, growth_rate)

        # Concatenate old and new features: (batch, height, width, channels) grows by growth_rate.
        # @arch denselayer.__call__.features-x-y:start
        features = [x, y]
        # @arch denselayer.__call__.features-x-y:end
        # @arch denselayer.__call__.y-jnp-concatenate-features-axis-n:start
        y = jnp.concatenate(features, axis=-1)  # (batch, height, width, channels) -> (batch, height, width, channels + growth_rate)
        # @arch denselayer.__call__.y-jnp-concatenate-features-axis-n:end
        # @arch denselayer.__call__.return-y:start
        return y
        # @arch denselayer.__call__.return-y:end


# %% [notebook-only]
# Create and run one dense layer: (2, 8, 8, 6) -> (2, 8, 8, 10).
layer = DenseLayer(growth_rate=4)
example_dense_input = jnp.ones((2, 8, 8, 6))  # -> (2, 8, 8, 6)
variables = layer.init(jax.random.PRNGKey(0), example_dense_input, train=False)
example_dense_output = layer.apply(variables, example_dense_input, train=False)  # (2, 8, 8, 6) -> (2, 8, 8, 10)
print("dense_output shape:", example_dense_output.shape)

# %%
# @arch class-transition-nn-module:start
class Transition(nn.Module):
# @arch class-transition-nn-module:end
    out_channels: int

    @nn.compact
    def __call__(self, x, train=False):
        # Compress channels and halve spatial resolution.
        # @arch transition.__call__.y-nn-batchnorm-use_running_average-not-train-name-norm-x:start
        y = nn.BatchNorm(use_running_average=not train, name='norm')(x)  # (batch, height, width, in_channels)
        # @arch transition.__call__.y-nn-batchnorm-use_running_average-not-train-name-norm-x:end
        # @arch transition.__call__.y-nn-relu-y:start
        y = nn.relu(y)  # (batch, height, width, in_channels)
        # @arch transition.__call__.y-nn-relu-y:end
        # @arch transition.__call__.y-nn-conv:start
        y = nn.Conv(
        # @arch transition.__call__.y-nn-conv:end
            # @arch transition.__call__.self-out_channels:start
            self.out_channels,
            # @arch transition.__call__.self-out_channels:end
            # @arch transition.__call__.n-n:start
            (1, 1),
            # @arch transition.__call__.n-n:end
            # @arch transition.__call__.use_bias-false:start
            use_bias=False,
            # @arch transition.__call__.use_bias-false:end
            # @arch transition.__call__.name-conv:start
            name='conv',
            # @arch transition.__call__.name-conv:end
        # @arch transition.__call__.y:start
        )(y)  # (batch, height, width, in_channels) -> (batch, height, width, out_channels)
        # @arch transition.__call__.y:end
        # @arch transition.__call__.y-nn-avg_pool-y-window_shape-n-n-strides-n-n-padding-valid:start
        y = nn.avg_pool(y, window_shape=(2, 2), strides=(2, 2), padding='VALID')  # (batch, height, width, out_channels) -> (batch, height/2, width/2, out_channels)
        # @arch transition.__call__.y-nn-avg_pool-y-window_shape-n-n-strides-n-n-padding-valid:end
        # @arch transition.__call__.return-y:start
        return y
        # @arch transition.__call__.return-y:end


# %% [notebook-only]
# Create and run one example_transition block: (2, 8, 8, 10) -> (2, 4, 4, 6).
example_transition = Transition(out_channels=6)
transition_input = jnp.ones((2, 8, 8, 10))  # -> (2, 8, 8, 10)
variables = example_transition.init(jax.random.PRNGKey(1), transition_input, train=False)
example_transition_output = example_transition.apply(variables, transition_input, train=False)  # (2, 8, 8, 10) -> (2, 4, 4, 6)
print("transition_output shape:", example_transition_output.shape)

# %%
# @arch class-densenet-nn-module:start
class DenseNet(nn.Module):
# @arch class-densenet-nn-module:end
    growth_rate: int = 32
    block_config: tuple = (6, 12, 24, 16)
    num_init_features: int = 64
    compression: float = 0.5
    dropout_rate: float = 0.0
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Convert image input into stem features: (batch, 224, 224, 3) -> (batch, 56, 56, 64).
        # @arch densenet.__call__.x-nn-conv:start
        x = nn.Conv(
        # @arch densenet.__call__.x-nn-conv:end
            # @arch densenet.__call__.self-num_init_features:start
            self.num_init_features,
            # @arch densenet.__call__.self-num_init_features:end
            # @arch densenet.__call__.n-n:start
            (7, 7),
            # @arch densenet.__call__.n-n:end
            # @arch densenet.__call__.strides-n-n:start
            strides=(2, 2),
            # @arch densenet.__call__.strides-n-n:end
            # @arch densenet.__call__.padding-same:start
            padding='SAME',
            # @arch densenet.__call__.padding-same:end
            # @arch densenet.__call__.use_bias-false:start
            use_bias=False,
            # @arch densenet.__call__.use_bias-false:end
            # @arch densenet.__call__.name-stem_conv:start
            name='stem_conv',
            # @arch densenet.__call__.name-stem_conv:end
        # @arch densenet.__call__.x:start
        )(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, num_init_features)
        # @arch densenet.__call__.x:end
        # @arch densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_norm-x:start
        x = nn.BatchNorm(use_running_average=not train, name='stem_norm')(x)  # (batch, 112, 112, num_init_features)
        # @arch densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-stem_norm-x:end
        # @arch densenet.__call__.x-nn-relu-x:start
        x = nn.relu(x)  # (batch, 112, 112, num_init_features)
        # @arch densenet.__call__.x-nn-relu-x:end
        # @arch densenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 112, 112, num_init_features) -> (batch, 56, 56, num_init_features)
        # @arch densenet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n-padding-same:end

        # Grow and compress feature maps through dense blocks and transitions.
        # @arch densenet.__call__.num_features-self-num_init_features:start
        num_features = self.num_init_features
        # @arch densenet.__call__.num_features-self-num_init_features:end
        # @arch densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config:start
        for block_index, layer_count in enumerate(self.block_config):
        # @arch densenet.__call__.for-block_index-layer_count-in-enumerate-self-block_config:end
            # @arch densenet.__call__.for-layer_index-in-range-layer_count:start
            for layer_index in range(layer_count):
            # @arch densenet.__call__.for-layer_index-in-range-layer_count:end
                # @arch densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n:start
                layer_name = f'denseblock{block_index + 1}.layer{layer_index + 1}'
                # @arch densenet.__call__.layer_name-f-denseblock-block_index-n-layer-layer_index-n:end
                # @arch densenet.__call__.x-denselayer:start
                x = DenseLayer(
                # @arch densenet.__call__.x-denselayer:end
                    # @arch densenet.__call__.self-growth_rate:start
                    self.growth_rate,
                    # @arch densenet.__call__.self-growth_rate:end
                    # @arch densenet.__call__.dropout_rate-self-dropout_rate:start
                    dropout_rate=self.dropout_rate,
                    # @arch densenet.__call__.dropout_rate-self-dropout_rate:end
                    # @arch densenet.__call__.name-layer_name:start
                    name=layer_name,
                    # @arch densenet.__call__.name-layer_name:end
                # @arch densenet.__call__.x-train-train:start
                )(x, train=train)  # (batch, height, width, channels) -> (batch, height, width, channels + growth_rate)
                # @arch densenet.__call__.x-train-train:end
                # @arch densenet.__call__.num_features-num_features-self-growth_rate:start
                num_features = num_features + self.growth_rate
                # @arch densenet.__call__.num_features-num_features-self-growth_rate:end

            # @arch densenet.__call__.is_last_block-block_index-len-self-block_config-n:start
            is_last_block = block_index == len(self.block_config) - 1
            # @arch densenet.__call__.is_last_block-block_index-len-self-block_config-n:end
            # @arch densenet.__call__.if-not-is_last_block:start
            if not is_last_block:
            # @arch densenet.__call__.if-not-is_last_block:end
                # @arch densenet.__call__.num_features-int-num_features-self-compression:start
                num_features = int(num_features * self.compression)
                # @arch densenet.__call__.num_features-int-num_features-self-compression:end
                # @arch densenet.__call__.transition_name-f-transition-block_index-n:start
                transition_name = f'transition{block_index + 1}'
                # @arch densenet.__call__.transition_name-f-transition-block_index-n:end
                # @arch densenet.__call__.x-transition-num_features-name-transition_name-x-train-train:start
                x = Transition(num_features, name=transition_name)(x, train=train)  # (batch, height, width, channels) -> (batch, height/2, width/2, num_features)
                # @arch densenet.__call__.x-transition-num_features-name-transition_name-x-train-train:end

        # Normalize, pool, and classify final dense features.
        # @arch densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-norm-x:start
        x = nn.BatchNorm(use_running_average=not train, name='norm')(x)  # (batch, 7, 7, num_features)
        # @arch densenet.__call__.x-nn-batchnorm-use_running_average-not-train-name-norm-x:end
        # @arch densenet.__call__.x-nn-relu-x.2:start
        x = nn.relu(x)  # (batch, 7, 7, num_features)
        # @arch densenet.__call__.x-nn-relu-x.2:end
        # @arch densenet.__call__.x-jnp-mean-x-axis-n-n:start
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, num_features) -> (batch, num_features)
        # @arch densenet.__call__.x-jnp-mean-x-axis-n-n:end
        # @arch densenet.__call__.logits-nn-dense-self-num_classes-name-classifier-x:start
        logits = nn.Dense(self.num_classes, name='classifier')(x)  # (batch, num_features) -> (batch, num_classes)
        # @arch densenet.__call__.logits-nn-dense-self-num_classes-name-classifier-x:end
        # @arch densenet.__call__.return-logits:start
        return logits
        # @arch densenet.__call__.return-logits:end


# %% [notebook-only]
# Create and run a sample ImageNet-size batch: (2, 224, 224, 3) -> (2, 1000).
example_model = DenseNet(num_classes=1000)
example_test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input, train=False)
example_logits = example_model.apply(example_params, example_test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = DenseNet(
    growth_rate=4,
    block_config=(1, 1, 1, 1),
    num_init_features=8,
    num_classes=2,
)
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
