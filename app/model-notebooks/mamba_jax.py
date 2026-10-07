# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class MambaBlock(nn.Module):
    model_dim: int = 32
    state_dim: int = 16
    expansion: int = 2
    kernel_size: int = 4

    @nn.compact
    def __call__(self, x):
        inner_dim = self.expansion * self.model_dim
        dt_rank = max(1, (self.model_dim + 15) // 16)

        # Normalize each token independently: (batch, steps, model_dim).
        # @arch mamba.block.norm:start
        norm_weight = self.param("norm_weight", nn.initializers.ones, (self.model_dim,))
        mean_square = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        inverse_rms = jax.lax.rsqrt(mean_square + 1e-5)
        normalized = x * inverse_rms
        normalized = normalized * norm_weight
        # @arch mamba.block.norm:end

        # Expand features into a state branch and a gate branch.
        # @arch mamba.block.project:start
        in_proj = nn.Dense(2 * inner_dim, use_bias=False, name="in_proj")
        projected = in_proj(normalized)  # -> (batch, steps, 2 * inner_dim)
        signal, gate = jnp.split(projected, 2, axis=-1)  # -> two (batch, steps, inner_dim) tensors
        # @arch mamba.block.project:end

        # Apply a depthwise convolution using only current and earlier tokens.
        # @arch mamba.block.conv:start
        conv = nn.Conv(
            inner_dim, kernel_size=(self.kernel_size,),
            padding=((self.kernel_size - 1, 0),), feature_group_count=inner_dim, name="conv"
        )
        signal = conv(signal)  # (batch, steps, inner_dim)
        signal = nn.silu(signal)  # (batch, steps, inner_dim)
        # @arch mamba.block.conv:end

        # Predict token-dependent time steps and state input/output vectors.
        # @arch mamba.block.select:start
        select_proj = nn.Dense(dt_rank + 2 * self.state_dim, use_bias=False, name="select_proj")
        selection = select_proj(signal)  # -> (batch, steps, dt_rank + 2 * state_dim)
        delta_low_rank, input_vector, output_vector = jnp.split(
            selection, [dt_rank, dt_rank + self.state_dim], axis=-1
        )
        # @arch mamba.block.select:end
        # @arch mamba.block.delta:start
        def dt_bias_init(key, shape, dtype=jnp.float32):
            time_steps = jnp.logspace(-3, -1, shape[0], dtype=dtype)
            inverse_softplus = time_steps + jnp.log(-jnp.expm1(-time_steps))
            return inverse_softplus

        dt_proj = nn.Dense(inner_dim, bias_init=dt_bias_init, name="dt_proj")
        delta_pre = dt_proj(delta_low_rank)  # (batch, steps, dt_rank) -> (batch, steps, inner_dim)
        delta = nn.softplus(delta_pre)  # (batch, steps, inner_dim)
        # @arch mamba.block.delta:end

        # Discretize the stable diagonal transition for each token.
        # @arch mamba.block.discretize:start
        def log_rate_init(key, shape):
            rates = jnp.arange(1, self.state_dim + 1, dtype=jnp.float32)
            log_rates = jnp.log(rates)
            return jnp.broadcast_to(log_rates, shape)

        A_log = self.param("A_log", log_rate_init, (inner_dim, self.state_dim))
        transition = -jnp.exp(A_log)  # (inner_dim, state_dim)
        delta_expanded = delta[..., None]  # -> (batch, steps, inner_dim, 1)
        discrete_transition = jnp.exp(delta_expanded * transition)  # -> (batch, steps, inner_dim, state_dim)
        input_expanded = input_vector[:, :, None, :]  # -> (batch, steps, 1, state_dim)
        discrete_input = delta_expanded * input_expanded  # -> (batch, steps, inner_dim, state_dim)
        # @arch mamba.block.discretize:end

        # Unroll the selective recurrence; this readable reference scan uses no fused GPU kernel.
        # @arch mamba.block.scan:start
        state = jnp.zeros((signal.shape[0], inner_dim, self.state_dim), dtype=signal.dtype)
        state_outputs = []
        for step in range(signal.shape[1]):
            current_input = signal[:, step, :, None]  # -> (batch, inner_dim, 1)
            retained_state = discrete_transition[:, step] * state  # (batch, inner_dim, state_dim)
            written_state = discrete_input[:, step] * current_input  # (batch, inner_dim, state_dim)
            state = retained_state + written_state  # (batch, inner_dim, state_dim)
            current_output_vector = output_vector[:, step, None, :]  # -> (batch, 1, state_dim)
            read_state = state * current_output_vector  # (batch, inner_dim, state_dim)
            state_output = jnp.sum(read_state, axis=-1)  # -> (batch, inner_dim)
            state_outputs.append(state_output)
        outputs = jnp.stack(state_outputs, axis=1)  # -> (batch, steps, inner_dim)
        # @arch mamba.block.scan:end

        # Add the learned direct path and modulate it with the SiLU gate.
        # @arch mamba.block.skip:start
        D = self.param("D", nn.initializers.ones, (inner_dim,))
        direct = signal * D  # (batch, steps, inner_dim)
        outputs = outputs + direct  # (batch, steps, inner_dim)
        # @arch mamba.block.skip:end
        # @arch mamba.block.gate:start
        activated_gate = nn.silu(gate)  # (batch, steps, inner_dim)
        gated_outputs = outputs * activated_gate  # (batch, steps, inner_dim)
        # @arch mamba.block.gate:end

        # Project back to the model width and preserve the residual stream.
        # @arch mamba.block.output:start
        out_proj = nn.Dense(self.model_dim, use_bias=False, name="out_proj")
        mixed = out_proj(gated_outputs)  # -> (batch, steps, model_dim)
        # @arch mamba.block.output:end
        # @arch mamba.block.residual:start
        outputs = x + mixed  # (batch, steps, model_dim)
        # @arch mamba.block.residual:end
        return outputs


# %% [notebook-only]
example_block = MambaBlock(model_dim=16, state_dim=8)
example_inputs = jnp.ones((2, 8, 16))  # -> (2, 8, 16)
example_params = example_block.init(jax.random.PRNGKey(0), example_inputs)
example_outputs = example_block.apply(example_params, example_inputs)  # (2, 8, 16)
print("Mamba block:", example_inputs.shape, "->", example_outputs.shape)


# %%
class MambaLM(nn.Module):
    vocab_size: int = 32
    model_dim: int = 32
    state_dim: int = 16
    layer_count: int = 2

    @nn.compact
    def __call__(self, tokens):
        # Consume token ids through the learned embedding table.
        embedding = nn.Embed(self.vocab_size, self.model_dim, name="embedding")
        # @arch mamba.embedding:start
        hidden = embedding(tokens)  # (batch, steps) -> (batch, steps, model_dim)
        # @arch mamba.embedding:end
        for index in range(self.layer_count):
            block = MambaBlock(self.model_dim, self.state_dim, name=f"block_{index}")
            hidden = block(hidden)  # (batch, steps, model_dim)

        # Normalize the final residual stream before vocabulary prediction.
        # @arch mamba.final_norm:start
        norm_weight = self.param("final_norm_weight", nn.initializers.ones, (self.model_dim,))
        mean_square = jnp.mean(jnp.square(hidden), axis=-1, keepdims=True)
        inverse_rms = jax.lax.rsqrt(mean_square + 1e-5)
        normalized = hidden * inverse_rms
        normalized = normalized * norm_weight
        # @arch mamba.final_norm:end

        # Reuse the embedding weights as the vocabulary readout.
        # @arch mamba.logits:start
        logits = embedding.attend(normalized)  # -> (batch, steps, vocab_size)
        # @arch mamba.logits:end
        return logits


# %% [notebook-only]
example_model = MambaLM(vocab_size=16, model_dim=16, state_dim=8)
example_tokens = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
example_params = example_model.init(jax.random.PRNGKey(0), example_tokens)
example_logits = example_model.apply(example_params, example_tokens)  # -> (2, 4, 16)
print("Mamba language model:", example_tokens.shape, "->", example_logits.shape)


# %%
# Fit next-token predictions on a tiny synthetic batch.
model = MambaLM(vocab_size=16, model_dim=16, state_dim=8)
train_tokens = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
train_targets = jnp.array([[2, 3, 4, 5], [3, 2, 1, 0]])  # -> (2, 4)
params = model.init(jax.random.PRNGKey(1), train_tokens)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # -> (batch, steps, vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1)
        loss = -jnp.mean(target_probs)  # -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


for step in range(3):
    params, loss = train_step(params, train_tokens, train_targets)
final_loss = loss  # scalar
