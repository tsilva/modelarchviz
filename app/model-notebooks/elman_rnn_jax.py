# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class ElmanRNN(nn.Module):
    hidden_size: int = 64
    output_size: int = 10

    @nn.compact
    def __call__(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        # @arch elmanrnn.__call__.batch_size-x-shape-n:start
        batch_size = x.shape[0]  # (batch, steps, input_size) -> scalar
        # @arch elmanrnn.__call__.batch_size-x-shape-n:end
        # @arch elmanrnn.__call__.hidden_shape-batch_size-self-hidden_size:start
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        # @arch elmanrnn.__call__.hidden_shape-batch_size-self-hidden_size:end
        # @arch elmanrnn.__call__.h-jnp-zeros-hidden_shape:start
        h = jnp.zeros(hidden_shape)  # -> (batch, hidden_size)
        # @arch elmanrnn.__call__.h-jnp-zeros-hidden_shape:end

        # Create shared projections for the recurrent loop.
        states = []
        # @arch elmanrnn.__call__.input_to_hidden-nn-dense-self-hidden_size-name-input_to_hidden:start
        input_to_hidden = nn.Dense(self.hidden_size, name='input_to_hidden')
        # @arch elmanrnn.__call__.input_to_hidden-nn-dense-self-hidden_size-name-input_to_hidden:end
        # @arch elmanrnn.__call__.hidden_to_hidden-nn-dense-self-hidden_size-use_bias-false-name-hidden_to:start
        hidden_to_hidden = nn.Dense(self.hidden_size, use_bias=False, name='hidden_to_hidden')
        # @arch elmanrnn.__call__.hidden_to_hidden-nn-dense-self-hidden_size-use_bias-false-name-hidden_to:end
        # @arch elmanrnn.__call__.hidden_to_output-nn-dense-self-output_size-name-hidden_to_output:start
        hidden_to_output = nn.Dense(self.output_size, name='hidden_to_output')
        # @arch elmanrnn.__call__.hidden_to_output-nn-dense-self-output_size-name-hidden_to_output:end

        # Run the shared recurrent cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        # @arch elmanrnn.__call__.step_count-x-shape-n:start
        step_count = x.shape[1]  # (batch, steps, input_size) -> scalar
        # @arch elmanrnn.__call__.step_count-x-shape-n:end
        # @arch elmanrnn.__call__.for-t-in-range-step_count:start
        for t in range(step_count):
        # @arch elmanrnn.__call__.for-t-in-range-step_count:end
            # @arch elmanrnn.__call__.current_input-x-t:start
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            # @arch elmanrnn.__call__.current_input-x-t:end
            # @arch elmanrnn.__call__.input_hidden-input_to_hidden-current_input:start
            input_hidden = input_to_hidden(current_input)  # (batch, input_size) -> (batch, hidden_size)
            # @arch elmanrnn.__call__.input_hidden-input_to_hidden-current_input:end
            # @arch elmanrnn.__call__.recurrent_hidden-hidden_to_hidden-h:start
            recurrent_hidden = hidden_to_hidden(h)  # (batch, hidden_size)
            # @arch elmanrnn.__call__.recurrent_hidden-hidden_to_hidden-h:end
            # @arch elmanrnn.__call__.hidden_sum-input_hidden-recurrent_hidden:start
            hidden_sum = input_hidden + recurrent_hidden  # (batch, hidden_size)
            # @arch elmanrnn.__call__.hidden_sum-input_hidden-recurrent_hidden:end
            # @arch elmanrnn.__call__.h-jnp-tanh-hidden_sum:start
            h = jnp.tanh(hidden_sum)  # (batch, hidden_size)
            # @arch elmanrnn.__call__.h-jnp-tanh-hidden_sum:end
            # @arch elmanrnn.__call__.states-append-h:start
            states.append(h)
            # @arch elmanrnn.__call__.states-append-h:end

        # Project the final state and pack the full state trace.
        # @arch elmanrnn.__call__.logits-hidden_to_output-h:start
        logits = hidden_to_output(h)  # (batch, hidden_size) -> (batch, output_size)
        # @arch elmanrnn.__call__.logits-hidden_to_output-h:end
        # @arch elmanrnn.__call__.state_trace-jnp-stack-states-axis-n:start
        state_trace = jnp.stack(states, axis=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        # @arch elmanrnn.__call__.state_trace-jnp-stack-states-axis-n:end
        # @arch elmanrnn.__call__.outputs-logits-state_trace:start
        outputs = (logits, state_trace)
        # @arch elmanrnn.__call__.outputs-logits-state_trace:end
        # @arch elmanrnn.__call__.return-outputs:start
        return outputs
        # @arch elmanrnn.__call__.return-outputs:end


# %% [notebook-only]
# Create and run a sample sequence: (2, 8, 32) -> logits and states.
example_model = ElmanRNN(hidden_size=64, output_size=10)
example_sequence = jnp.ones((2, 8, 32))  # -> (2, 8, 32)
example_params = example_model.init(jax.random.PRNGKey(0), example_sequence)
example_outputs = example_model.apply(example_params, example_sequence)
example_logits = example_outputs[0]  # (2, 10)
example_states = example_outputs[1]  # (2, 8, 64)
print("logits shape:", example_logits.shape)

# %%
# Train on two synthetic sequences with opposite labels.
model = ElmanRNN(hidden_size=8, output_size=2)
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
