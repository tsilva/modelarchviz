# Create and run a squeeze-excitation gate: (2, 16, 16, 8) -> (2, 16, 16, 8).
gate = SqueezeExcite(squeeze_channels=2)
feature_map = jnp.ones((2, 16, 16, 8))  # -> (2, 16, 16, 8)
params = gate.init(jax.random.PRNGKey(0), feature_map)
gated = gate.apply(params, feature_map)  # (2, 16, 16, 8) -> (2, 16, 16, 8)

# Create and run one mobile inverted bottleneck: (2, 16, 16, 8) -> (2, 16, 16, 8).
block = MBConv(out_channels=8, expand_ratio=1, stride=1, kernel_size=3)
block_input = jnp.ones((2, 16, 16, 8))  # -> (2, 16, 16, 8)
variables = block.init(jax.random.PRNGKey(1), block_input, train=False)
block_output = block.apply(variables, block_input, train=False)  # (2, 16, 16, 8) -> (2, 16, 16, 8)

# Create and run a sample ImageNet-size batch: (2, 224, 224, 3) -> (2, 1000).
model = EfficientNet(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
params = model.init(jax.random.PRNGKey(0), test_input, train=False)
logits = model.apply(params, test_input, train=False)  # (2, 224, 224, 3) -> (2, 1000)


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
