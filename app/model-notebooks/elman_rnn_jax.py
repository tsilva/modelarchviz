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


class ElmanRNN(nn.Module):
    hidden_size: int = 64
    output_size: int = 10

    @nn.compact
    def __call__(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        batch_size = x.shape[0]
        hidden_shape = (batch_size, self.hidden_size)
        h = jnp.zeros(hidden_shape)

        # Create shared projections for the recurrent loop.
        states = []
        input_to_hidden = nn.Dense(self.hidden_size, name='input_to_hidden')
        hidden_to_hidden = nn.Dense(self.hidden_size, use_bias=False, name='hidden_to_hidden')
        hidden_to_output = nn.Dense(self.output_size, name='hidden_to_output')

        # Run the shared recurrent cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        step_count = x.shape[1]
        for t in range(step_count):
            current_input = x[:, t]
            input_hidden = input_to_hidden(current_input)
            recurrent_hidden = hidden_to_hidden(h)
            hidden_sum = input_hidden + recurrent_hidden
            h = jnp.tanh(hidden_sum)
            states.append(h)

        # Project the final state and pack the full state trace.
        logits = hidden_to_output(h)
        state_trace = jnp.stack(states, axis=1)
        outputs = (logits, state_trace)
        return outputs


# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = ElmanRNN(hidden_size=64, output_size=10)
sequence = jnp.ones((2, 8, 32))
params = model.init(jax.random.PRNGKey(0), sequence)
outputs = model.apply(params, sequence)
logits = outputs[0]
states = outputs[1]

# logits: (2, 10), states: (2, 8, 64)
