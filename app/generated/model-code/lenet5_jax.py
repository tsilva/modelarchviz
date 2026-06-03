import jax
import jax.numpy as jnp
from flax import linen as nn


class LeNet5(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Extract convolutional features: (batch, 32, 32, 1) -> (batch, 5, 5, 16).
        conv1 = nn.Conv(features=6, kernel_size=(5, 5))(x)
        x = jnp.tanh(conv1)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        conv2 = nn.Conv(features=16, kernel_size=(5, 5))(x)
        x = jnp.tanh(conv2)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten feature maps for dense layers: (batch, 5, 5, 16) -> (batch, 400).
        batch_size = x.shape[0]
        flat_shape = (batch_size, -1)
        x = x.reshape(flat_shape)

        # Classify flattened features: (batch, 400) -> (batch, 10).
        fc1 = nn.Dense(features=120)(x)
        x = jnp.tanh(fc1)
        fc2 = nn.Dense(features=84)(x)
        x = jnp.tanh(fc2)
        logits = nn.Dense(features=10)(x)
        return logits


# Create and run a sample image batch: (2, 32, 32, 1) -> (2, 10).
model = LeNet5()
test_input = jnp.ones((2, 32, 32, 1))
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)

# logits: (2, 10)

# Train on a tiny synthetic image batch.
model = LeNet5()
train_images = jnp.zeros((2, 32, 32, 1))
train_images = train_images.at[0, 8:16, 8:16, :].set(1.0)
train_images = train_images.at[1, 16:24, 16:24, :].set(1.0)
train_targets = jnp.array([0, 1])
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

final_loss = loss
