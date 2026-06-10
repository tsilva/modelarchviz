# Create and run a sample image batch: (2, 32, 32, 1) -> (2, 10).
model = LeNet5()
test_input = jnp.ones((2, 32, 32, 1))  # -> (2, 32, 32, 1)
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)  # (2, 32, 32, 1) -> (2, 10)


# Train on a tiny synthetic image batch.
model = LeNet5()
train_images = jnp.zeros((2, 32, 32, 1))  # -> (2, 32, 32, 1)
train_images = train_images.at[0, 8:16, 8:16, :].set(1.0)  # (2, 32, 32, 1)
train_images = train_images.at[1, 16:24, 16:24, :].set(1.0)  # (2, 32, 32, 1)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # (batch, 32, 32, 1) -> (batch, 10)
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
