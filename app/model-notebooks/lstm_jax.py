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
class LSTMCell(nn.Module):
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x, state):
        # Unpack recurrent state: tuple -> two (batch, hidden_size) tensors.
        h, c = state  # ((batch, hidden_size), (batch, hidden_size))

        # Compute input gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_i = nn.Dense(self.hidden_size, name='x_i')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_i = nn.Dense(self.hidden_size, use_bias=False, name='h_i')(h)  # (batch, hidden_size)
        i_pre = x_i + h_i  # (batch, hidden_size)
        i = nn.sigmoid(i_pre)  # (batch, hidden_size)

        # Compute forget gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_f = nn.Dense(self.hidden_size, name='x_f')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_f = nn.Dense(self.hidden_size, use_bias=False, name='h_f')(h)  # (batch, hidden_size)
        f_pre = x_f + h_f  # (batch, hidden_size)
        f = nn.sigmoid(f_pre)  # (batch, hidden_size)

        # Compute candidate memory: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_g = nn.Dense(self.hidden_size, name='x_g')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_g = nn.Dense(self.hidden_size, use_bias=False, name='h_g')(h)  # (batch, hidden_size)
        g_pre = x_g + h_g  # (batch, hidden_size)
        g = jnp.tanh(g_pre)  # (batch, hidden_size)

        # Compute output gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_o = nn.Dense(self.hidden_size, name='x_o')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_o = nn.Dense(self.hidden_size, use_bias=False, name='h_o')(h)  # (batch, hidden_size)
        o_pre = x_o + h_o  # (batch, hidden_size)
        o = nn.sigmoid(o_pre)  # (batch, hidden_size)

        # Blend previous memory with candidate memory: (batch, hidden_size).
        forget_c = f * c  # (batch, hidden_size)
        write_c = i * g  # (batch, hidden_size)
        c_next = forget_c + write_c  # (batch, hidden_size)

        # Read hidden state from updated memory: (batch, hidden_size).
        c_readout = jnp.tanh(c_next)  # (batch, hidden_size)
        h_next = o * c_readout  # (batch, hidden_size)
        next_state = (h_next, c_next)
        return next_state


# %% [notebook-only]
# Create and run one LSTM cell step: (2, 32), state -> next state.
example_cell = LSTMCell(hidden_size=64)
example_input = jnp.ones((2, 32))  # -> (2, 32)
example_previous_state = (
    jnp.zeros((2, 64)),
    jnp.zeros((2, 64)),
)
example_params = example_cell.init(jax.random.PRNGKey(0), example_input, example_previous_state)
example_next_state = example_cell.apply(example_params, example_input, example_previous_state)
example_hidden = example_next_state[0]  # tuple -> (2, 64)
example_cell_state = example_next_state[1]  # tuple -> (2, 64)
print("next hidden shape:", example_hidden.shape, "next example_cell shape:", example_cell_state.shape)


# %%
class LSTMSequence(nn.Module):
    hidden_size: int = 64
    output_size: int = 10

    @nn.compact
    def __call__(self, x):
        # Build initial recurrent state: (batch, hidden_size).
        batch_size = x.shape[0]  # (batch, steps, input_size) -> scalar
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = jnp.zeros(hidden_shape)  # -> (batch, hidden_size)
        c = jnp.zeros(hidden_shape)  # -> (batch, hidden_size)

        # Run the shared LSTM cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        cell = LSTMCell(self.hidden_size)
        step_count = x.shape[1]  # (batch, steps, input_size) -> scalar
        for t in range(step_count):
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            previous_state = (h, c)
            next_state = cell(current_input, previous_state)
            h = next_state[0]  # (batch, hidden_size)
            c = next_state[1]  # (batch, hidden_size)
            states.append(h)

        # Project the final hidden state and pack the full state trace.
        logits = nn.Dense(self.output_size, name='readout')(h)  # (batch, hidden_size) -> (batch, output_size)
        state_trace = jnp.stack(states, axis=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        outputs = (logits, state_trace)
        return outputs


# %% [notebook-only]
# Create and run a sample sequence: (2, 8, 32) -> logits and states.
example_model = LSTMSequence(hidden_size=64, output_size=10)
example_sequence = jnp.ones((2, 8, 32))  # -> (2, 8, 32)
example_params = example_model.init(jax.random.PRNGKey(0), example_sequence)
example_outputs = example_model.apply(example_params, example_sequence)
example_logits = example_outputs[0]  # (2, 10)
example_states = example_outputs[1]  # (2, 8, 64)
print("example_logits shape:", example_logits.shape, "example_states shape:", example_states.shape)


# %%
# Train on two synthetic sequences with opposite labels.
model = LSTMSequence(hidden_size=64, output_size=10)
train_sequences = jnp.zeros((2, 3, 32))  # -> (2, 3, 32)
train_sequences = train_sequences.at[0, :, 0].set(jnp.array([1.0, 0.5, 1.0]))  # (3)
train_sequences = train_sequences.at[1, :, 1].set(jnp.array([1.0, 0.5, 1.0]))  # (3)
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
