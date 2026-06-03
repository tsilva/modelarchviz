import jax
import jax.numpy as jnp
from flax import linen as nn


class GRUCell(nn.Module):
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x, h):
        # Compute update gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_z = nn.Dense(self.hidden_size, name='x_z')(x)
        h_z = nn.Dense(self.hidden_size, use_bias=False, name='h_z')(h)
        z_pre = x_z + h_z
        z = nn.sigmoid(z_pre)

        # Compute reset gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_r = nn.Dense(self.hidden_size, name='x_r')(x)
        h_r = nn.Dense(self.hidden_size, use_bias=False, name='h_r')(h)
        r_pre = x_r + h_r
        r = nn.sigmoid(r_pre)

        # Compute candidate state from reset hidden state: (batch, hidden_size).
        reset_h = r * h
        x_n = nn.Dense(self.hidden_size, name='x_n')(x)
        h_n = nn.Dense(self.hidden_size, use_bias=False, name='h_n')(reset_h)
        n_pre = x_n + h_n
        n = jnp.tanh(n_pre)

        # Blend previous and candidate states: (batch, hidden_size).
        keep_h = z * h
        candidate_h = (1.0 - z) * n
        h_next = candidate_h + keep_h
        return h_next


class GRUSequence(nn.Module):
    hidden_size: int = 64
    output_size: int = 10

    @nn.compact
    def __call__(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        batch_size = x.shape[0]
        hidden_shape = (batch_size, self.hidden_size)
        h = jnp.zeros(hidden_shape)

        # Run the shared GRU cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        cell = GRUCell(self.hidden_size)
        step_count = x.shape[1]
        for t in range(step_count):
            current_input = x[:, t]
            h = cell(current_input, h)
            states.append(h)

        # Project the final hidden state and pack the full state trace.
        logits = nn.Dense(self.output_size, name='readout')(h)
        state_trace = jnp.stack(states, axis=1)
        outputs = (logits, state_trace)
        return outputs


# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = GRUSequence(hidden_size=64, output_size=10)
sequence = jnp.ones((2, 8, 32))
params = model.init(jax.random.PRNGKey(0), sequence)
outputs = model.apply(params, sequence)
logits = outputs[0]
states = outputs[1]


# Train on two synthetic sequences with opposite labels.
model = GRUSequence(hidden_size=8, output_size=2)
train_sequences = jnp.array(
    [
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
    ]
)
train_targets = jnp.array([0, 1])
params = model.init(jax.random.PRNGKey(1), train_sequences)


def train_step(params, inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        outputs = model.apply(current_params, inputs)
        logits = outputs[0]
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_sequences, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss
