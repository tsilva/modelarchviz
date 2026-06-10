# Create and run one Inception block: (2, 16, 16, 8) -> (2, 16, 16, 16).
block = InceptionBlock(4, 4, 4, 4, 4, 4)
block_input = jnp.ones((2, 16, 16, 8))  # -> (2, 16, 16, 8)
params = block.init(jax.random.PRNGKey(0), block_input)
block_output = block.apply(params, block_input)  # (2, 16, 16, 8) -> (2, 16, 16, 16)

# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
model = GoogLeNet(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)


# Train on a tiny synthetic image batch.
model = GoogLeNet(num_classes=2)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images, train=False)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs, train=False)  # (batch, 224, 224, 3) -> (batch, num_classes)
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
