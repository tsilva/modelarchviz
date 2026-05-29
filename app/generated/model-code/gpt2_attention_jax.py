import jax
import jax.numpy as jnp
from flax import linen as nn


class GPT2Small(nn.Module):
    vocab_size: int
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    @nn.compact
    def __call__(self, input_ids, mask):
        # Combine token and position embeddings: (batch, steps) -> (batch, steps, n_embd).
        batch_size, step_count = input_ids.shape
        positions = jnp.arange(step_count)
        token_embeddings = nn.Embed(self.vocab_size, self.n_embd, name='wte')(input_ids)
        position_embeddings = nn.Embed(self.n_ctx, self.n_embd, name='wpe')(positions)
        position_embeddings = position_embeddings[None, :, :]
        x = token_embeddings + position_embeddings
        x = nn.Dropout(0.1, deterministic=True, name='drop')(x)

        # Run the transformer block stack while preserving sequence shape.
        for _ in range(self.n_layer):
            x = Block()(x, mask)

        # Normalize final states and project to vocabulary logits.
        x = nn.LayerNorm(name='ln_f')(x)
        logits = nn.Dense(self.vocab_size, name='lm_head')(x)
        return logits


class CausalSelfAttention(nn.Module):
    n_embd: int = 768
    n_head: int = 12

    def __call__(self, x, mask):
        # Project hidden states into query, key, and value tensors.
        batch_size, step_count, channel_count = x.shape
        qkv_dim = 3 * channel_count
        qkv = nn.Dense(qkv_dim, name='c_attn')(x)
        q, k, v = jnp.split(qkv, 3, axis=2)
        head_dim = channel_count // self.n_head

        # Split heads: (batch, steps, channels) -> (batch, heads, steps, head_dim).
        q = q.reshape(batch_size, step_count, self.n_head, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, step_count, self.n_head, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, step_count, self.n_head, head_dim)
        v = v.transpose(0, 2, 1, 3)

        # Compute masked causal attention weights.
        key_transpose = jnp.swapaxes(k, -2, -1)
        scores = q @ key_transpose
        scale = k.shape[-1] ** -0.5
        att = scores * scale
        mask_window = mask[:, :, :step_count, :step_count]
        att = jnp.where(mask_window == 0, -jnp.inf, att)
        weights = nn.softmax(att, axis=-1)

        # Merge heads back to the model width and project.
        y = weights @ v
        y = y.transpose(0, 2, 1, 3)
        y = y.reshape(batch_size, step_count, channel_count)
        out = nn.Dense(channel_count, name='c_proj')(y)
        return out


class MLP(nn.Module):
    n_embd: int = 768
    hidden_dim: int = 3072

    @nn.compact
    def __call__(self, x):
        # Expand and contract hidden states through the GPT feed-forward block.
        hidden = nn.Dense(self.hidden_dim, name='c_fc')(x)
        hidden = nn.gelu(hidden)
        out = nn.Dense(self.n_embd, name='c_proj')(hidden)
        return out


class Block(nn.Module):
    def __call__(self, x, mask):
        # Apply causal attention with a residual connection.
        attn_input = nn.LayerNorm(name='ln_1')(x)
        attn = CausalSelfAttention()(attn_input, mask)
        x = x + attn

        # Apply MLP with a residual connection.
        mlp_input = nn.LayerNorm(name='ln_2')(x)
        mlp_out = MLP()(mlp_input)
        x = x + mlp_out
        return x


# Create and run a sample token batch.
model = GPT2Small(vocab_size=50257)
test_input = jnp.ones((2, 16), dtype=jnp.int32)

# Build a causal attention mask: (1, 1, 16, 16).
mask_values = jnp.ones((16, 16))
mask = jnp.tril(mask_values)
mask = mask.reshape(1, 1, 16, 16)
params = model.init(jax.random.PRNGKey(0), test_input, mask)
logits = model.apply(params, test_input, mask)

# logits: (2, 16, 50257)
