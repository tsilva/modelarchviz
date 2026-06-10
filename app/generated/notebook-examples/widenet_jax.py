# Create and run one widened residual block: (2, 32, 32, 16) -> (2, 16, 16, 32).
block = WideBasicBlock(out_channels=32, stride=2, dropout_rate=0.0)
block_input = jnp.ones((2, 32, 32, 16))  # -> (2, 32, 32, 16)
variables = block.init(jax.random.PRNGKey(0), block_input, train=False)
block_output = block.apply(variables, block_input, train=False)  # (2, 32, 32, 16) -> (2, 16, 16, 32)

# Create and run a sample CIFAR-size image batch: (2, 32, 32, 3) -> (2, 10).
model = WideNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=10)
test_input = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)  # (2, 32, 32, 3) -> (2, 10)


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
