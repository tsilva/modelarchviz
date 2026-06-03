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

# Train on two synthetic sequences with opposite labels.
model = ElmanRNN(hidden_size=8, output_size=2)
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


for step in range(3):
    params, loss = train_step(params, train_sequences, train_targets)

final_loss = loss
