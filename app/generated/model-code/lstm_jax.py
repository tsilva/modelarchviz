import jax
import jax.numpy as jnp
from flax import linen as nn

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
