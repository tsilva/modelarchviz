# Create and run one double convolution: (2, 32, 32, 1) -> (2, 32, 32, 8).
block = DoubleConv(out_channels=8)
block_input = jnp.ones((2, 32, 32, 1))  # -> (2, 32, 32, 1)
params = block.init(jax.random.PRNGKey(0), block_input)
block_output = block.apply(params, block_input)  # (2, 32, 32, 1) -> (2, 32, 32, 8)

# Create and run a sample image batch: (2, 572, 572, 1) -> (2, 572, 572, 2).
model = UNet(num_classes=2)
test_input = jnp.ones((2, 572, 572, 1))  # -> (2, 572, 572, 1)
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)  # (2, 572, 572, 1) -> (2, 572, 572, 2)


# Train on two synthetic segmentation masks.
model = UNet(num_classes=2)
train_images = jnp.zeros((2, 64, 64, 1))  # -> (2, 64, 64, 1)
train_images = train_images.at[0, 8:32, 8:32, :].set(1.0)  # (2, 64, 64, 1)
train_images = train_images.at[1, 32:56, 32:56, :].set(1.0)  # (2, 64, 64, 1)
train_targets = jnp.zeros((2, 64, 64), dtype=jnp.int32)  # -> (2, 64, 64)
train_targets = train_targets.at[0, 8:32, 8:32].set(1)
train_targets = train_targets.at[1, 32:56, 32:56].set(1)
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # (batch, height, width, 1) -> (batch, height, width, num_classes)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch, height, width) -> (batch, height, width, num_classes)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, height, width, num_classes)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, height, width, num_classes) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
