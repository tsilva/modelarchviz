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
