# Create and run one GRU cell step: (2, 32), (2, 64) -> (2, 64).
cell = GRUCell(hidden_size=64)
cell_input = jnp.ones((2, 32))  # -> (2, 32)
previous_state = jnp.zeros((2, 64))  # -> (2, 64)
cell_params = cell.init(jax.random.PRNGKey(0), cell_input, previous_state)
next_state = cell.apply(cell_params, cell_input, previous_state)  # (2, 32), (2, 64) -> (2, 64)

# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = GRUSequence(hidden_size=64, output_size=10)
sequence = jnp.ones((2, 8, 32))  # -> (2, 8, 32)
params = model.init(jax.random.PRNGKey(1), sequence)
outputs = model.apply(params, sequence)
logits = outputs[0]  # (2, 10)
states = outputs[1]  # (2, 8, 64)

# Train the same model on two synthetic sequences with opposite labels.
train_sequences = jnp.zeros((2, 3, 32))  # -> (2, 3, 32)
first_pattern = jnp.array([1.0, 0.5, 1.0])  # -> (3)
second_pattern = jnp.array([1.0, 0.5, 1.0])  # -> (3)
train_sequences = train_sequences.at[0, :, 0].set(first_pattern)  # (2, 3, 32)
train_sequences = train_sequences.at[1, :, 1].set(second_pattern)  # (2, 3, 32)
train_targets = jnp.array([0, 1])  # -> (2)


def train_step(params, inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        outputs = model.apply(current_params, inputs)
        logits = outputs[0]  # (batch, output_size)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, output_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, output_size)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, output_size) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_sequences, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
