# Create and run one LSTM cell step: (2, 32), state -> next state.
cell = LSTMCell(hidden_size=64)
cell_input = jnp.ones((2, 32))  # -> (2, 32)
previous_state = (
    jnp.zeros((2, 64)),
    jnp.zeros((2, 64)),
)
params = cell.init(jax.random.PRNGKey(0), cell_input, previous_state)
next_state = cell.apply(params, cell_input, previous_state)
next_hidden = next_state[0]  # tuple -> (2, 64)
next_cell = next_state[1]  # tuple -> (2, 64)

# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = LSTMSequence(hidden_size=64, output_size=10)
sequence = jnp.ones((2, 8, 32))  # -> (2, 8, 32)
params = model.init(jax.random.PRNGKey(0), sequence)
outputs = model.apply(params, sequence)
logits = outputs[0]  # (2, 10)
states = outputs[1]  # (2, 8, 64)


# Train on two synthetic sequences with opposite labels.
model = LSTMSequence(hidden_size=8, output_size=2)
train_sequences = jnp.array(
    [
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
    ]
)  # -> (2, 3, 3)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_sequences)


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
