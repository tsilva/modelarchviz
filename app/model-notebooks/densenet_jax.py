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
class DenseLayer(nn.Module):
    growth_rate: int
    bottleneck_width: int = 4
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train=False):
        # Compress existing features, then produce new growth features.
        bottleneck_channels = self.bottleneck_width * self.growth_rate
        y = nn.BatchNorm(use_running_average=not train, name='norm1')(x)  # (batch, height, width, channels)
        y = nn.relu(y)  # (batch, height, width, channels)
        y = nn.Conv(
            bottleneck_channels,
            (1, 1),
            use_bias=False,
            name='conv1',
        )(y)  # (batch, height, width, channels) -> (batch, height, width, bottleneck_channels)
        y = nn.BatchNorm(use_running_average=not train, name='norm2')(y)  # (batch, height, width, bottleneck_channels)
        y = nn.relu(y)  # (batch, height, width, bottleneck_channels)
        y = nn.Conv(
            self.growth_rate,
            (3, 3),
            padding='SAME',
            use_bias=False,
            name='conv2',
        )(y)  # (batch, height, width, bottleneck_channels) -> (batch, height, width, growth_rate)
        if self.dropout_rate > 0:
            y = nn.Dropout(
                rate=self.dropout_rate,
                name='dropout',
            )(y, deterministic=not train)  # (batch, height, width, growth_rate)

        # Concatenate old and new features: (batch, height, width, channels) grows by growth_rate.
        features = [x, y]
        y = jnp.concatenate(features, axis=-1)  # (batch, height, width, channels) -> (batch, height, width, channels + growth_rate)
        return y


# %% [notebook-only]
# Create and run one dense layer: (2, 8, 8, 6) -> (2, 8, 8, 10).
layer = DenseLayer(growth_rate=4)
example_dense_input = jnp.ones((2, 8, 8, 6))  # -> (2, 8, 8, 6)
variables = layer.init(jax.random.PRNGKey(0), example_dense_input, train=False)
example_dense_output = layer.apply(variables, example_dense_input, train=False)  # (2, 8, 8, 6) -> (2, 8, 8, 10)
print("dense_output shape:", example_dense_output.shape)

# %%
class Transition(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, train=False):
        # Compress channels and halve spatial resolution.
        y = nn.BatchNorm(use_running_average=not train, name='norm')(x)  # (batch, height, width, in_channels)
        y = nn.relu(y)  # (batch, height, width, in_channels)
        y = nn.Conv(
            self.out_channels,
            (1, 1),
            use_bias=False,
            name='conv',
        )(y)  # (batch, height, width, in_channels) -> (batch, height, width, out_channels)
        y = nn.avg_pool(y, window_shape=(2, 2), strides=(2, 2), padding='VALID')  # (batch, height, width, out_channels) -> (batch, height/2, width/2, out_channels)
        return y


# %% [notebook-only]
# Create and run one example_transition block: (2, 8, 8, 10) -> (2, 4, 4, 6).
example_transition = Transition(out_channels=6)
transition_input = jnp.ones((2, 8, 8, 10))  # -> (2, 8, 8, 10)
variables = example_transition.init(jax.random.PRNGKey(1), transition_input, train=False)
example_transition_output = example_transition.apply(variables, transition_input, train=False)  # (2, 8, 8, 10) -> (2, 4, 4, 6)
print("transition_output shape:", example_transition_output.shape)

# %%
class DenseNet(nn.Module):
    growth_rate: int = 32
    block_config: tuple = (6, 12, 24, 16)
    num_init_features: int = 64
    compression: float = 0.5
    dropout_rate: float = 0.0
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Convert image input into stem features: (batch, 224, 224, 3) -> (batch, 56, 56, 64).
        x = nn.Conv(
            self.num_init_features,
            (7, 7),
            strides=(2, 2),
            padding='SAME',
            use_bias=False,
            name='stem_conv',
        )(x)  # (batch, 224, 224, 3) -> (batch, 112, 112, num_init_features)
        x = nn.BatchNorm(use_running_average=not train, name='stem_norm')(x)  # (batch, 112, 112, num_init_features)
        x = nn.relu(x)  # (batch, 112, 112, num_init_features)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')  # (batch, 112, 112, num_init_features) -> (batch, 56, 56, num_init_features)

        # Grow and compress feature maps through dense blocks and transitions.
        num_features = self.num_init_features
        for block_index, layer_count in enumerate(self.block_config):
            for layer_index in range(layer_count):
                layer_name = f'denseblock{block_index + 1}.layer{layer_index + 1}'
                x = DenseLayer(
                    self.growth_rate,
                    dropout_rate=self.dropout_rate,
                    name=layer_name,
                )(x, train=train)  # (batch, height, width, channels) -> (batch, height, width, channels + growth_rate)
                num_features = num_features + self.growth_rate

            is_last_block = block_index == len(self.block_config) - 1
            if not is_last_block:
                num_features = int(num_features * self.compression)
                transition_name = f'transition{block_index + 1}'
                x = Transition(num_features, name=transition_name)(x, train=train)  # (batch, height, width, channels) -> (batch, height/2, width/2, num_features)

        # Normalize, pool, and classify final dense features.
        x = nn.BatchNorm(use_running_average=not train, name='norm')(x)  # (batch, 7, 7, num_features)
        x = nn.relu(x)  # (batch, 7, 7, num_features)
        x = jnp.mean(x, axis=(1, 2))  # (batch, 7, 7, num_features) -> (batch, num_features)
        logits = nn.Dense(self.num_classes, name='classifier')(x)  # (batch, num_features) -> (batch, num_classes)
        return logits


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
