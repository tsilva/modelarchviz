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


class PositionalEncoding(nn.Module):
    d_model: int = 512

    @nn.compact
    def __call__(self, x):
        # Build sinusoidal position encodings for the current sequence length.
        seq_len = x.shape[1]  # (batch, steps, d_model) -> scalar
        positions = jnp.arange(seq_len)  # -> (steps)
        position = positions[:, None]  # (steps) -> (steps, 1)
        even_indices = jnp.arange(0, self.d_model, 2)  # -> (d_model // 2)
        scale = -jnp.log(10000.0) / self.d_model  # scalar -> scalar
        div_term = jnp.exp(even_indices * scale)  # (d_model // 2) -> (d_model // 2)
        pe = jnp.zeros((seq_len, self.d_model))  # -> (steps, d_model)
        sin_values = jnp.sin(position * div_term)  # (steps, 1), (d_model // 2) -> (steps, d_model // 2)
        cos_values = jnp.cos(position * div_term)  # (steps, 1), (d_model // 2) -> (steps, d_model // 2)
        pe = pe.at[:, 0::2].set(sin_values)  # (steps, d_model), (steps, d_model // 2) -> (steps, d_model)
        pe = pe.at[:, 1::2].set(cos_values)  # (steps, d_model), (steps, d_model // 2) -> (steps, d_model)
        batch_pe = pe[None, :, :]  # (steps, d_model) -> (1, steps, d_model)

        # Add position encodings to embeddings: (batch, steps, d_model).
        encoded = x + batch_pe  # (batch, steps, d_model), (1, steps, d_model) -> (batch, steps, d_model)
        return encoded  # (batch, steps, d_model)


class EncoderLayer(nn.Module):
    d_model: int = 512
    nhead: int = 8
    d_ff: int = 2048

    @nn.compact
    def __call__(self, x):
        # Apply self-attention with residual normalization: (batch, steps, d_model).
        attn = nn.MultiHeadDotProductAttention(num_heads=self.nhead)(x, x)  # (batch, steps, d_model) -> (batch, steps, d_model)
        attn_residual = x + attn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        x = nn.LayerNorm()(attn_residual)  # (batch, steps, d_model) -> (batch, steps, d_model)

        # Apply feed-forward block with residual normalization.
        ffn_layers = [nn.Dense(self.d_ff), nn.relu, nn.Dense(self.d_model)]
        ffn = nn.Sequential(ffn_layers)(x)  # (batch, steps, d_model) -> (batch, steps, d_model)
        ffn_residual = x + ffn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        out = nn.LayerNorm()(ffn_residual)  # (batch, steps, d_model) -> (batch, steps, d_model)
        return out  # (batch, steps, d_model)


class DecoderLayer(nn.Module):
    d_model: int = 512
    nhead: int = 8
    d_ff: int = 2048

    @nn.compact
    def __call__(self, x, memory, mask):
        # Apply masked self-attention with residual normalization.
        masked = nn.MultiHeadDotProductAttention(num_heads=self.nhead)(x, x, mask=mask)  # (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        masked_residual = x + masked  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        x = nn.LayerNorm()(masked_residual)  # (batch, target_steps, d_model) -> (batch, target_steps, d_model)

        # Attend over encoder memory with residual normalization.
        cross = nn.MultiHeadDotProductAttention(num_heads=self.nhead)(x, memory)  # (batch, target_steps, d_model), (batch, source_steps, d_model) -> (batch, target_steps, d_model)
        cross_residual = x + cross  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        x = nn.LayerNorm()(cross_residual)  # (batch, target_steps, d_model) -> (batch, target_steps, d_model)

        # Apply feed-forward block with residual normalization.
        ffn_layers = [nn.Dense(self.d_ff), nn.relu, nn.Dense(self.d_model)]
        ffn = nn.Sequential(ffn_layers)(x)  # (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        ffn_residual = x + ffn  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        out = nn.LayerNorm()(ffn_residual)  # (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        return out  # (batch, target_steps, d_model)


class Transformer(nn.Module):
    vocab_size: int = 37000
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6

    @nn.compact
    def __call__(self, src_ids, tgt_ids, tgt_mask):
        # Embed and encode the source tokens: (batch, source_steps) -> memory.
        src_embedding = nn.Embed(self.vocab_size, self.d_model)(src_ids)  # (batch, source_steps) -> (batch, source_steps, d_model)
        memory = PositionalEncoding(self.d_model)(src_embedding)  # (batch, source_steps, d_model) -> (batch, source_steps, d_model)
        for _ in range(self.num_layers):
            memory = EncoderLayer(self.d_model, self.nhead)(memory)  # (batch, source_steps, d_model) -> (batch, source_steps, d_model)

        # Embed target tokens and decode against source memory.
        tgt_embedding = nn.Embed(self.vocab_size, self.d_model)(tgt_ids)  # (batch, target_steps) -> (batch, target_steps, d_model)
        x = PositionalEncoding(self.d_model)(tgt_embedding)  # (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        for _ in range(self.num_layers):
            x = DecoderLayer(self.d_model, self.nhead)(x, memory, tgt_mask)  # (batch, target_steps, d_model) -> (batch, target_steps, d_model)

        # Project decoder states to vocabulary logits.
        logits = nn.Dense(self.vocab_size)(x)  # (batch, target_steps, d_model) -> (batch, target_steps, vocab_size)
        return logits  # (batch, target_steps, vocab_size)


# Create and run a sample translation batch.
model = Transformer(vocab_size=37000)
src_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)
tgt_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)

# Build a causal target mask: (1, 1, 16, 16).
mask_values = jnp.ones((1, 1, 16, 16))  # -> (1, 1, 16, 16)
tgt_mask = jnp.tril(mask_values)  # (1, 1, 16, 16) -> (1, 1, 16, 16)
params = model.init(jax.random.PRNGKey(0), src_ids, tgt_ids, tgt_mask)  # inputs -> parameter tree
logits = model.apply(params, src_ids, tgt_ids, tgt_mask)  # (2, 16), (2, 16), (1, 1, 16, 16) -> (2, 16, 37000)

# Train on a tiny copy-style token batch.
model = Transformer(vocab_size=20, d_model=16, nhead=4, num_layers=1)
src_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
tgt_ids = jnp.array([[0, 1, 2, 3], [0, 4, 3, 2]], dtype=jnp.int32)  # -> (2, 4)
train_targets = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
mask_values = jnp.ones((1, 1, 4, 4))  # -> (1, 1, 4, 4)
tgt_mask = jnp.tril(mask_values)  # (1, 1, 4, 4) -> (1, 1, 4, 4)
params = model.init(jax.random.PRNGKey(1), src_ids, tgt_ids, tgt_mask)  # inputs -> parameter tree


def train_step(params, src_ids, tgt_ids, targets, mask, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, src_ids, tgt_ids, mask)  # (2, 4), (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 4, 20) -> (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)  # parameter tree -> scalar, gradient tree
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)  # parameter tree -> parameter tree
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, src_ids, tgt_ids, train_targets, tgt_mask)  # parameter tree -> parameter tree, scalar

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar -> scalar
