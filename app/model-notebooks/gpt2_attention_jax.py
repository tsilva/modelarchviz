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
class GPT2Small(nn.Module):
    vocab_size: int
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    @nn.compact
    def __call__(self, input_ids, mask):
        # Combine token and position embeddings: (batch, steps) -> (batch, steps, n_embd).
        batch_size, step_count = input_ids.shape  # (batch, steps) -> scalar, scalar
        positions = jnp.arange(step_count)  # -> (steps)
        token_embeddings = nn.Embed(self.vocab_size, self.n_embd, name='wte')(input_ids)  # (batch, steps) -> (batch, steps, n_embd)
        position_embeddings = nn.Embed(self.n_ctx, self.n_embd, name='wpe')(positions)  # (steps) -> (steps, n_embd)
        position_embeddings = position_embeddings[None, :, :]  # (steps, n_embd) -> (1, steps, n_embd)
        x = token_embeddings + position_embeddings  # (batch, steps, n_embd)
        x = nn.Dropout(0.1, deterministic=True, name='drop')(x)  # (batch, steps, n_embd)

        # Run the transformer block stack while preserving sequence shape.
        for _ in range(self.n_layer):
            x = Block()(x, mask)  # (batch, steps, n_embd)

        # Normalize final states and project to vocabulary logits.
        x = nn.LayerNorm(name='ln_f')(x)  # (batch, steps, n_embd)
        logits = nn.Dense(self.vocab_size, name='lm_head')(x)  # (batch, steps, n_embd) -> (batch, steps, vocab_size)
        return logits  # (batch, steps, vocab_size)


# %%
class CausalSelfAttention(nn.Module):
    n_embd: int = 768
    n_head: int = 12

    @nn.compact
    def __call__(self, x, mask):
        # Project hidden states into query, key, and value tensors.
        batch_size, step_count, channel_count = x.shape  # (batch, steps, channels) -> scalar, scalar, scalar
        qkv_dim = 3 * channel_count  # scalar
        qkv = nn.Dense(qkv_dim, name='c_attn')(x)  # (batch, steps, channels) -> (batch, steps, 3 * channels)
        q, k, v = jnp.split(qkv, 3, axis=2)  # (batch, steps, 3 * channels) -> three (batch, steps, channels)
        head_dim = channel_count // self.n_head  # scalar

        # Split heads: (batch, steps, channels) -> (batch, heads, steps, head_dim).
        q = q.reshape(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        q = q.transpose(0, 2, 1, 3)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        k = k.reshape(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        v = v.reshape(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)

        # Compute masked causal attention weights.
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        scale = k.shape[-1] ** -0.5  # (batch, heads, steps, head_dim) -> scalar
        att = scores * scale  # (batch, heads, steps, steps)
        mask_window = mask[:, :, :step_count, :step_count]  # (1, 1, max_steps, max_steps) -> (1, 1, steps, steps)
        att = jnp.where(mask_window == 0, -jnp.inf, att)  # (batch, heads, steps, steps)
        weights = nn.softmax(att, axis=-1)  # (batch, heads, steps, steps)

        # Merge heads back to the model width and project.
        y = weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        y = y.transpose(0, 2, 1, 3)  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        y = y.reshape(batch_size, step_count, channel_count)  # (batch, steps, heads, head_dim) -> (batch, steps, channels)
        out = nn.Dense(channel_count, name='c_proj')(y)  # (batch, steps, channels)
        return out  # (batch, steps, channels)


# %% [notebook-only]
# Create and run causal self-attention: (2, 4, 24) -> (2, 4, 24).
attention = CausalSelfAttention(n_embd=24, n_head=4)
hidden_states = jnp.ones((2, 4, 24))  # -> (2, 4, 24)
mask = jnp.tril(jnp.ones((4, 4))).reshape(1, 1, 4, 4)  # -> (1, 1, 4, 4)
params = attention.init(jax.random.PRNGKey(0), hidden_states, mask)
attended = attention.apply(params, hidden_states, mask)  # (2, 4, 24), (1, 1, 4, 4) -> (2, 4, 24)


# %%
class MLP(nn.Module):
    n_embd: int = 768
    hidden_dim: int = 3072

    @nn.compact
    def __call__(self, x):
        # Expand and contract hidden states through the GPT feed-forward block.
        hidden = nn.Dense(self.hidden_dim, name='c_fc')(x)  # (batch, steps, n_embd) -> (batch, steps, hidden_dim)
        hidden = nn.gelu(hidden)  # (batch, steps, hidden_dim)
        out = nn.Dense(self.n_embd, name='c_proj')(hidden)  # (batch, steps, hidden_dim) -> (batch, steps, n_embd)
        return out  # (batch, steps, n_embd)


# %% [notebook-only]
# Create and run the GPT feed-forward block: (2, 4, 24) -> (2, 4, 24).
mlp = MLP(n_embd=24, hidden_dim=48)
hidden_states = jnp.ones((2, 4, 24))  # -> (2, 4, 24)
params = mlp.init(jax.random.PRNGKey(1), hidden_states)
mlp_output = mlp.apply(params, hidden_states)  # (2, 4, 24) -> (2, 4, 24)


# %%
class Block(nn.Module):
    @nn.compact
    def __call__(self, x, mask):
        # Apply causal attention with a residual connection.
        attn_input = nn.LayerNorm(name='ln_1')(x)  # (batch, steps, 768)
        attn = CausalSelfAttention()(attn_input, mask)  # (batch, steps, 768)
        x = x + attn  # (batch, steps, 768)

        # Apply MLP with a residual connection.
        mlp_input = nn.LayerNorm(name='ln_2')(x)  # (batch, steps, 768)
        mlp_out = MLP()(mlp_input)  # (batch, steps, 768)
        x = x + mlp_out  # (batch, steps, 768)
        return x  # (batch, steps, 768)


# %% [notebook-only]
# Create and run a sample token batch.
model = GPT2Small(vocab_size=50257)
test_input = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)

# Build a causal attention mask: (1, 1, 16, 16).
mask_values = jnp.ones((16, 16))  # -> (16, 16)
mask = jnp.tril(mask_values)  # (16, 16)
mask = mask.reshape(1, 1, 16, 16)  # (16, 16) -> (1, 1, 16, 16)
params = model.init(jax.random.PRNGKey(0), test_input, mask)
logits = model.apply(params, test_input, mask)  # (2, 16), (1, 1, 16, 16) -> (2, 16, 50257)


# Train on a tiny next-token prediction batch.
model = GPT2Small(vocab_size=20, n_layer=1)
input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
train_targets = jnp.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=jnp.int32)  # -> (2, 4)
mask_values = jnp.ones((4, 4))  # -> (4, 4)
mask = jnp.tril(mask_values)  # (4, 4)
mask = mask.reshape(1, 1, 4, 4)  # (4, 4) -> (1, 1, 4, 4)
params = model.init(jax.random.PRNGKey(1), input_ids, mask)


def train_step(params, inputs, mask, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs, mask)  # (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, input_ids, mask, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
