# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# Define a multilayer perceptron.
class MLP(nn.Module):
    # @arch mlp.hidden_dim-int-n:start
    hidden_dim: int = 128
    # @arch mlp.hidden_dim-int-n:end
    # @arch mlp.output_dim-int-n:start
    output_dim: int = 10
    # @arch mlp.output_dim-int-n:end

    @nn.compact
    def __call__(self, x):
        # First hidden block: (batch, features) -> (batch, hidden_dim).
        # @arch mlp.__call__.hiddenn-nn-dense-self-hidden_dim-name-hiddenn:start
        hidden1 = nn.Dense(self.hidden_dim, name='hidden1')
        # @arch mlp.__call__.hiddenn-nn-dense-self-hidden_dim-name-hiddenn:end
        # @arch mlp.__call__.hn_pre-hiddenn-x:start
        h1_pre = hidden1(x)  # (batch, features) -> (batch, hidden_dim)
        # @arch mlp.__call__.hn_pre-hiddenn-x:end
        # @arch mlp.__call__.hn-nn-sigmoid-hn_pre:start
        h1 = nn.sigmoid(h1_pre)  # (batch, hidden_dim)
        # @arch mlp.__call__.hn-nn-sigmoid-hn_pre:end

        # Second hidden block keeps the hidden shape: (batch, hidden_dim).
        # @arch mlp.__call__.hiddenn-nn-dense-self-hidden_dim-name-hiddenn.2:start
        hidden2 = nn.Dense(self.hidden_dim, name='hidden2')
        # @arch mlp.__call__.hiddenn-nn-dense-self-hidden_dim-name-hiddenn.2:end
        # @arch mlp.__call__.hn_pre-hiddenn-hn:start
        h2_pre = hidden2(h1)  # (batch, hidden_dim)
        # @arch mlp.__call__.hn_pre-hiddenn-hn:end
        # @arch mlp.__call__.hn-nn-sigmoid-hn_pre.2:start
        h2 = nn.sigmoid(h2_pre)  # (batch, hidden_dim)
        # @arch mlp.__call__.hn-nn-sigmoid-hn_pre.2:end

        # Output block: (batch, hidden_dim) -> (batch, output_dim).
        # @arch mlp.__call__.output-nn-dense-self-output_dim-name-output:start
        output = nn.Dense(self.output_dim, name='output')
        # @arch mlp.__call__.output-nn-dense-self-output_dim-name-output:end
        # @arch mlp.__call__.logits-output-hn:start
        logits = output(h2)  # (batch, hidden_dim) -> (batch, output_dim)
        # @arch mlp.__call__.logits-output-hn:end
        # @arch mlp.__call__.return-logits:start
        return logits
        # @arch mlp.__call__.return-logits:end


# %% [notebook-only]
# Create and run a sample batch: (2, 784) -> (2, 10).
example_model = MLP(hidden_dim=128, output_dim=10)
inputs = jnp.ones((2, 784))  # -> (2, 784)
example_params = example_model.init(jax.random.PRNGKey(0), inputs)
example_logits = example_model.apply(example_params, inputs)  # (2, 784) -> (2, 10)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic classification batch.
model = MLP(hidden_dim=8, output_dim=2)
# @arch train_inputs-jnp-array:start
train_inputs = jnp.array(
# @arch train_inputs-jnp-array:end
    # @arch code.4:start
    [
    # @arch code.4:end
        # @arch n-n-n-n:start
        [1.0, 0.0, 1.0, 0.0],
        # @arch n-n-n-n:end
        # @arch n-n-n-n.2:start
        [0.0, 1.0, 0.0, 1.0],
        # @arch n-n-n-n.2:end
    # @arch code.5:start
    ]
    # @arch code.5:end
# @arch code.6:start
)  # -> (2, 4)
# @arch code.6:end
train_targets = jnp.array([0, 1])  # -> (2)
# @arch params-model-init-jax-random-prngkey-n-train_inputs:start
params = model.init(jax.random.PRNGKey(1), train_inputs)
# @arch params-model-init-jax-random-prngkey-n-train_inputs:end


def train_step(params, inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        # @arch train_step.loss_fn.logits-model-apply-current_params-inputs:start
        logits = model.apply(current_params, inputs)  # (batch, features) -> (batch, output_dim)
        # @arch train_step.loss_fn.logits-model-apply-current_params-inputs:end
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch) -> (batch, output_dim)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, output_dim)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (batch, output_dim) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_inputs, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
