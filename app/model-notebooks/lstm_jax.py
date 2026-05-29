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


class LSTMCell(nn.Module):
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x, state):
        # Unpack recurrent state: tuple -> two (batch, hidden_size) tensors.
        h, c = state

        # Compute input gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_i = nn.Dense(self.hidden_size, name='x_i')(x)
        h_i = nn.Dense(self.hidden_size, use_bias=False, name='h_i')(h)
        i_pre = x_i + h_i
        i = nn.sigmoid(i_pre)

        # Compute forget gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_f = nn.Dense(self.hidden_size, name='x_f')(x)
        h_f = nn.Dense(self.hidden_size, use_bias=False, name='h_f')(h)
        f_pre = x_f + h_f
        f = nn.sigmoid(f_pre)

        # Compute candidate memory: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_g = nn.Dense(self.hidden_size, name='x_g')(x)
        h_g = nn.Dense(self.hidden_size, use_bias=False, name='h_g')(h)
        g_pre = x_g + h_g
        g = jnp.tanh(g_pre)

        # Compute output gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_o = nn.Dense(self.hidden_size, name='x_o')(x)
        h_o = nn.Dense(self.hidden_size, use_bias=False, name='h_o')(h)
        o_pre = x_o + h_o
        o = nn.sigmoid(o_pre)

        # Blend previous memory with candidate memory: (batch, hidden_size).
        forget_c = f * c
        write_c = i * g
        c_next = forget_c + write_c

        # Read hidden state from updated memory: (batch, hidden_size).
        c_readout = jnp.tanh(c_next)
        h_next = o * c_readout
        next_state = (h_next, c_next)
        return next_state


class LSTMSequence(nn.Module):
    hidden_size: int = 64
    output_size: int = 10

    @nn.compact
    def __call__(self, x):
        # Build initial recurrent state: (batch, hidden_size).
        batch_size = x.shape[0]
        hidden_shape = (batch_size, self.hidden_size)
        h = jnp.zeros(hidden_shape)
        c = jnp.zeros(hidden_shape)

        # Run the shared LSTM cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        cell = LSTMCell(self.hidden_size)
        step_count = x.shape[1]
        for t in range(step_count):
            current_input = x[:, t]
            previous_state = (h, c)
            next_state = cell(current_input, previous_state)
            h = next_state[0]
            c = next_state[1]
            states.append(h)

        # Project the final hidden state and pack the full state trace.
        logits = nn.Dense(self.output_size, name='readout')(h)
        state_trace = jnp.stack(states, axis=1)
        outputs = (logits, state_trace)
        return outputs


# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = LSTMSequence(hidden_size=64, output_size=10)
sequence = jnp.ones((2, 8, 32))
params = model.init(jax.random.PRNGKey(0), sequence)
outputs = model.apply(params, sequence)
logits = outputs[0]
states = outputs[1]

# logits: (2, 10), states: (2, 8, 64)
