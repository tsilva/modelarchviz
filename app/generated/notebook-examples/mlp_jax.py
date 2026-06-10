# Create and run a sample batch: (2, 784) -> (2, 10).
model = MLP(hidden_dim=128, output_dim=10)
inputs = jnp.ones((2, 784))  # -> (2, 784)
params = model.init(jax.random.PRNGKey(0), inputs)
logits = model.apply(params, inputs)  # (2, 784) -> (2, 10)


# Train on a tiny synthetic classification batch.
model = MLP(hidden_dim=8, output_dim=2)
train_inputs = jnp.array(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
)  # -> (2, 4)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_inputs)


def train_step(params, inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # (batch, features) -> (batch, output_dim)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, output_dim)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, output_dim)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, output_dim) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_inputs, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
