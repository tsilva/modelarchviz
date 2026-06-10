# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class GRUCell(nn.Module):
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x, h):
        # Compute update gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_z = nn.Dense(self.hidden_size, name='x_z')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_z = nn.Dense(self.hidden_size, use_bias=False, name='h_z')(h)  # (batch, hidden_size)
        z_pre = x_z + h_z  # (batch, hidden_size)
        z = nn.sigmoid(z_pre)  # (batch, hidden_size)

        # Compute reset gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_r = nn.Dense(self.hidden_size, name='x_r')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_r = nn.Dense(self.hidden_size, use_bias=False, name='h_r')(h)  # (batch, hidden_size)
        r_pre = x_r + h_r  # (batch, hidden_size)
        r = nn.sigmoid(r_pre)  # (batch, hidden_size)

        # Compute candidate state from reset hidden state: (batch, hidden_size).
        reset_h = r * h  # (batch, hidden_size)
        x_n = nn.Dense(self.hidden_size, name='x_n')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_n = nn.Dense(self.hidden_size, use_bias=False, name='h_n')(reset_h)  # (batch, hidden_size)
        n_pre = x_n + h_n  # (batch, hidden_size)
        n = jnp.tanh(n_pre)  # (batch, hidden_size)

        # Blend previous and candidate states: (batch, hidden_size).
        keep_h = z * h  # (batch, hidden_size)
        candidate_h = (1.0 - z) * n  # (batch, hidden_size)
        h_next = candidate_h + keep_h  # (batch, hidden_size)
        return h_next


# %% [notebook-only]
# Create and run one GRU cell step: (2, 32), (2, 64) -> (2, 64).
example_cell = GRUCell(hidden_size=64)
example_cell_input = jnp.ones((2, 32))  # -> (2, 32)
example_previous_state = jnp.zeros((2, 64))  # -> (2, 64)
cell_params = example_cell.init(jax.random.PRNGKey(0), example_cell_input, example_previous_state)
example_next_state = example_cell.apply(cell_params, example_cell_input, example_previous_state)  # (2, 32), (2, 64) -> (2, 64)
print("example_next_state shape:", example_next_state.shape)


# %%
class GRUSequence(nn.Module):
    hidden_size: int = 64
    output_size: int = 10

    @nn.compact
    def __call__(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        batch_size = x.shape[0]  # (batch, steps, input_size) -> scalar
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = jnp.zeros(hidden_shape)  # -> (batch, hidden_size)

        # Run the shared GRU cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        cell = GRUCell(self.hidden_size)
        step_count = x.shape[1]  # (batch, steps, input_size) -> scalar
        for t in range(step_count):
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            h = cell(current_input, h)  # (batch, input_size), (batch, hidden_size) -> (batch, hidden_size)
            states.append(h)

        # Project the final hidden state and pack the full state trace.
        logits = nn.Dense(self.output_size, name='readout')(h)  # (batch, hidden_size) -> (batch, output_size)
        state_trace = jnp.stack(states, axis=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        outputs = (logits, state_trace)
        return outputs


# %% [notebook-only]
# Create and run a sample sequence: (2, 8, 32) -> logits and states.
example_model = GRUSequence(hidden_size=64, output_size=10)
example_sequence = jnp.ones((2, 8, 32))  # -> (2, 8, 32)
example_params = example_model.init(jax.random.PRNGKey(1), example_sequence)
example_outputs = example_model.apply(example_params, example_sequence)
example_logits = example_outputs[0]  # (2, 10)
example_states = example_outputs[1]  # (2, 8, 64)
print("example_logits shape:", example_logits.shape, "example_states shape:", example_states.shape)


# %%
# Train the same model on two synthetic sequences with opposite labels.
model = GRUSequence(hidden_size=64, output_size=10)
train_sequences = jnp.zeros((2, 3, 32))  # -> (2, 3, 32)
first_pattern = jnp.array([1.0, 0.5, 1.0])  # -> (3)
second_pattern = jnp.array([1.0, 0.5, 1.0])  # -> (3)
train_sequences = train_sequences.at[0, :, 0].set(first_pattern)  # (2, 3, 32)
train_sequences = train_sequences.at[1, :, 1].set(second_pattern)  # (2, 3, 32)
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
