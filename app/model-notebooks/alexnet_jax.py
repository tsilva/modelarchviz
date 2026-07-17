# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch def-local_response_norm-x-size-n-alpha-ne-n-beta-n-k-n:start
def local_response_norm(x, size=5, alpha=1e-4, beta=0.75, k=2.0):
# @arch def-local_response_norm-x-size-n-alpha-ne-n-beta-n-k-n:end
    # Build the local channel window used for response normalization.
    # @arch local_response_norm.half-size-n:start
    half = size // 2
    # @arch local_response_norm.half-size-n:end
    # @arch local_response_norm.squared-jnp-square-x:start
    squared = jnp.square(x)  # (batch, height, width, channels)
    # @arch local_response_norm.squared-jnp-square-x:end
    # @arch local_response_norm.padded-jnp-pad-squared-n-n-n-n-n-n-half-half:start
    padded = jnp.pad(squared, ((0, 0), (0, 0), (0, 0), (half, half)))  # (batch, height, width, channels) -> (batch, height, width, channels + 2 * half)
    # @arch local_response_norm.padded-jnp-pad-squared-n-n-n-n-n-n-half-half:end

    # Accumulate neighboring channel energy and normalize activations.
    # @arch local_response_norm.scale-k:start
    scale = k
    # @arch local_response_norm.scale-k:end
    # @arch local_response_norm.for-offset-in-range-size:start
    for offset in range(size):
    # @arch local_response_norm.for-offset-in-range-size:end
        # @arch local_response_norm.channel_end-offset-x-shape-n:start
        channel_end = offset + x.shape[-1]  # (batch, height, width, channels) -> scalar
        # @arch local_response_norm.channel_end-offset-x-shape-n:end
        # @arch local_response_norm.window-padded-offset-channel_end:start
        window = padded[..., offset:channel_end]  # (batch, height, width, channels + 2 * half) -> (batch, height, width, channels)
        # @arch local_response_norm.window-padded-offset-channel_end:end
        # @arch local_response_norm.scale_step-alpha-size-window:start
        scale_step = (alpha / size) * window  # (batch, height, width, channels)
        # @arch local_response_norm.scale_step-alpha-size-window:end
        # @arch local_response_norm.scale-scale-scale_step:start
        scale = scale + scale_step  # (batch, height, width, channels)
        # @arch local_response_norm.scale-scale-scale_step:end
    # @arch local_response_norm.denominator-jnp-power-scale-beta:start
    denominator = jnp.power(scale, beta)  # (batch, height, width, channels)
    # @arch local_response_norm.denominator-jnp-power-scale-beta:end
    # @arch local_response_norm.normalized-x-denominator:start
    normalized = x / denominator  # (batch, height, width, channels)
    # @arch local_response_norm.normalized-x-denominator:end
    # @arch local_response_norm.return-normalized:start
    return normalized
    # @arch local_response_norm.return-normalized:end


# %%
class AlexNet(nn.Module):
    num_classes: int = 1000

    @nn.compact
    def __call__(self, x, train=False):
        # Extract early high-stride features: (batch, 227, 227, 3) -> pooled feature maps.
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x:start
        x = nn.Conv(features=96, kernel_size=(11, 11), strides=(4, 4), name='conv1')(x)  # (batch, 227, 227, 3) -> (batch, 55, 55, 96)
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-strides-n-n-name-convn-x:end
        # @arch alexnet.__call__.x-nn-relu-x:start
        x = nn.relu(x)  # (batch, 55, 55, 96)
        # @arch alexnet.__call__.x-nn-relu-x:end
        # @arch alexnet.__call__.x-local_response_norm-x:start
        x = local_response_norm(x)  # (batch, 55, 55, 96)
        # @arch alexnet.__call__.x-local_response_norm-x:end
        # @arch alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # (batch, 55, 55, 96) -> (batch, 27, 27, 96)
        # @arch alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n:end

        # Refine convolutional features through middle and late AlexNet blocks.
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x:start
        x = nn.Conv(features=256, kernel_size=(5, 5), padding='SAME', name='conv2')(x)  # (batch, 27, 27, 96) -> (batch, 27, 27, 256)
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x:end
        # @arch alexnet.__call__.x-nn-relu-x.2:start
        x = nn.relu(x)  # (batch, 27, 27, 256)
        # @arch alexnet.__call__.x-nn-relu-x.2:end
        # @arch alexnet.__call__.x-local_response_norm-x.2:start
        x = local_response_norm(x)  # (batch, 27, 27, 256)
        # @arch alexnet.__call__.x-local_response_norm-x.2:end
        # @arch alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.2:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # (batch, 27, 27, 256) -> (batch, 13, 13, 256)
        # @arch alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.2:end
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.2:start
        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv3')(x)  # (batch, 13, 13, 256) -> (batch, 13, 13, 384)
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.2:end
        # @arch alexnet.__call__.x-nn-relu-x.3:start
        x = nn.relu(x)  # (batch, 13, 13, 384)
        # @arch alexnet.__call__.x-nn-relu-x.3:end
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.3:start
        x = nn.Conv(features=384, kernel_size=(3, 3), padding='SAME', name='conv4')(x)  # (batch, 13, 13, 384)
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.3:end
        # @arch alexnet.__call__.x-nn-relu-x.4:start
        x = nn.relu(x)  # (batch, 13, 13, 384)
        # @arch alexnet.__call__.x-nn-relu-x.4:end
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.4:start
        x = nn.Conv(features=256, kernel_size=(3, 3), padding='SAME', name='conv5')(x)  # (batch, 13, 13, 384) -> (batch, 13, 13, 256)
        # @arch alexnet.__call__.x-nn-conv-features-n-kernel_size-n-n-padding-same-name-convn-x.4:end
        # @arch alexnet.__call__.x-nn-relu-x.5:start
        x = nn.relu(x)  # (batch, 13, 13, 256)
        # @arch alexnet.__call__.x-nn-relu-x.5:end
        # @arch alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3:start
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))  # (batch, 13, 13, 256) -> (batch, 6, 6, 256)
        # @arch alexnet.__call__.x-nn-max_pool-x-window_shape-n-n-strides-n-n.3:end

        # Flatten feature maps for dense classification: (batch, spatial, spatial, channels) -> (batch, features).
        batch_size = x.shape[0]  # (batch, 6, 6, 256) -> scalar
        flat_shape = (batch_size, -1)  # -> (batch, features)
        # @arch alexnet.__call__.x-x-reshape-flat_shape:start
        x = x.reshape(flat_shape)  # (batch, 6, 6, 256) -> (batch, 9216)
        # @arch alexnet.__call__.x-x-reshape-flat_shape:end

        # Classify flattened features: (batch, features) -> (batch, num_classes).
        # @arch alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x:start
        x = nn.Dropout(0.5, deterministic=not train)(x)  # (batch, 9216)
        # @arch alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x:end
        # @arch alexnet.__call__.x-nn-dense-features-n-name-fcn-x:start
        x = nn.Dense(features=4096, name='fc6')(x)  # (batch, 9216) -> (batch, 4096)
        # @arch alexnet.__call__.x-nn-dense-features-n-name-fcn-x:end
        # @arch alexnet.__call__.x-nn-relu-x.6:start
        x = nn.relu(x)  # (batch, 4096)
        # @arch alexnet.__call__.x-nn-relu-x.6:end
        # @arch alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x.2:start
        x = nn.Dropout(0.5, deterministic=not train)(x)  # (batch, 4096)
        # @arch alexnet.__call__.x-nn-dropout-n-deterministic-not-train-x.2:end
        # @arch alexnet.__call__.x-nn-dense-features-n-name-fcn-x.2:start
        x = nn.Dense(features=4096, name='fc7')(x)  # (batch, 4096)
        # @arch alexnet.__call__.x-nn-dense-features-n-name-fcn-x.2:end
        # @arch alexnet.__call__.x-nn-relu-x.7:start
        x = nn.relu(x)  # (batch, 4096)
        # @arch alexnet.__call__.x-nn-relu-x.7:end
        # @arch alexnet.__call__.logits-nn-dense-features-self-num_classes-name-fcn-x:start
        logits = nn.Dense(features=self.num_classes, name='fc8')(x)  # (batch, 4096) -> (batch, num_classes)
        # @arch alexnet.__call__.logits-nn-dense-features-self-num_classes-name-fcn-x:end
        return logits


# %% [notebook-only]
# Create and run a sample image batch: (2, 227, 227, 3) -> (2, 1000).
example_model = AlexNet(num_classes=1000)
example_test_input = jnp.ones((2, 227, 227, 3))  # -> (2, 227, 227, 3)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input, train=False)
example_logits = example_model.apply(example_params, example_test_input, train=False)  # (2, 227, 227, 3) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = AlexNet(num_classes=2)
train_images = jnp.zeros((2, 227, 227, 3))  # -> (2, 227, 227, 3)
train_images = train_images.at[0, 40:100, 40:100, :].set(1.0)  # (2, 227, 227, 3)
train_images = train_images.at[1, 120:180, 120:180, :].set(1.0)  # (2, 227, 227, 3)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images, train=False)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs, train=False)  # (batch, 227, 227, 3) -> (batch, num_classes)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, classes) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss
