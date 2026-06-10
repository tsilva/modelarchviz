import jax
import jax.numpy as jnp
from flax import linen as nn

class PatchEmbed(nn.Module):
    embed_dim: int = 768
    patch_size: int = 16

    @nn.compact
    def __call__(self, x):
        # Project image patches: (batch, height, width, channels) -> (batch, grid, grid, embed_dim).
        projection = nn.Conv(self.embed_dim, (self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), name='proj')
        x = projection(x)  # (batch, height, width, channels) -> (batch, grid, grid, embed_dim)

        # Flatten patches into a token sequence: (batch, grid, grid, embed_dim) -> (batch, patches, embed_dim).
        batch_size = x.shape[0]  # (batch, grid, grid, embed_dim) -> scalar
        sequence_shape = (batch_size, -1, self.embed_dim)  # -> (batch, patches, embed_dim)
        x = x.reshape(sequence_shape)  # (batch, grid, grid, embed_dim) -> (batch, patches, embed_dim)
        return x  # (batch, patches, embed_dim)

class MultiHeadSelfAttention(nn.Module):
    embed_dim: int = 768
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        # Project tokens into per-head query, key, and value tensors.
        batch_size = x.shape[0]  # (batch, tokens, embed_dim) -> scalar
        tokens = x.shape[1]  # (batch, tokens, embed_dim) -> scalar
        head_dim = self.embed_dim // self.num_heads  # scalar
        q = nn.Dense(self.embed_dim, name='q_proj')(x)  # (batch, tokens, embed_dim)
        k = nn.Dense(self.embed_dim, name='k_proj')(x)  # (batch, tokens, embed_dim)
        v = nn.Dense(self.embed_dim, name='v_proj')(x)  # (batch, tokens, embed_dim)

        # Split model width across heads: (batch, tokens, embed_dim) -> (batch, heads, tokens, head_dim).
        head_shape = (batch_size, tokens, self.num_heads, head_dim)
        q = q.reshape(head_shape)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        k = k.reshape(head_shape)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        v = v.reshape(head_shape)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)

        # Compute scaled dot-product attention over all image tokens.
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, tokens, head_dim) -> (batch, heads, head_dim, tokens)
        scores = q @ key_transpose  # (batch, heads, tokens, head_dim), (batch, heads, head_dim, tokens) -> (batch, heads, tokens, tokens)
        scale = head_dim ** -0.5  # scalar
        attn_scores = scores * scale  # (batch, heads, tokens, tokens)
        attn_weights = nn.softmax(attn_scores, axis=-1)  # (batch, heads, tokens, tokens)

        # Mix values, merge heads, and project back to embedding width.
        context = attn_weights @ v  # (batch, heads, tokens, tokens), (batch, heads, tokens, head_dim) -> (batch, heads, tokens, head_dim)
        context = jnp.transpose(context, (0, 2, 1, 3))  # (batch, heads, tokens, head_dim) -> (batch, tokens, heads, head_dim)
        merged_shape = (batch_size, tokens, self.embed_dim)
        merged = context.reshape(merged_shape)  # (batch, tokens, heads, head_dim) -> (batch, tokens, embed_dim)
        out = nn.Dense(self.embed_dim, name='out_proj')(merged)  # (batch, tokens, embed_dim)
        return out  # (batch, tokens, embed_dim)

class EncoderBlock(nn.Module):
    embed_dim: int = 768
    num_heads: int = 12
    mlp_dim: int = 3072

    @nn.compact
    def __call__(self, x):
        # Apply self-attention with a residual connection.
        y = nn.LayerNorm(name='ln_1')(x)  # (batch, tokens, embed_dim)
        y = MultiHeadSelfAttention(self.embed_dim, self.num_heads)(y)  # (batch, tokens, embed_dim)
        x = x + y  # (batch, tokens, embed_dim)

        # Apply MLP with a residual connection.
        y = nn.LayerNorm(name='ln_2')(x)  # (batch, tokens, embed_dim)
        y = nn.Dense(self.mlp_dim, name='mlp_fc1')(y)  # (batch, tokens, embed_dim) -> (batch, tokens, mlp_dim)
        y = nn.gelu(y)  # (batch, tokens, mlp_dim)
        y = nn.Dense(self.embed_dim, name='mlp_fc2')(y)  # (batch, tokens, mlp_dim) -> (batch, tokens, embed_dim)
        out = x + y  # (batch, tokens, embed_dim)
        return out  # (batch, tokens, embed_dim)

class VisionTransformer(nn.Module):
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        # Convert image patches into tokens and prepend CLS token.
        x = PatchEmbed(self.embed_dim)(x)  # (batch, height, width, channels) -> (batch, patches, embed_dim)
        cls = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))  # -> (1, 1, embed_dim)
        batch_size = x.shape[0]  # (batch, patches, embed_dim) -> scalar
        cls_shape = (batch_size, 1, 1)  # -> (batch, 1, 1)
        cls = jnp.tile(cls, cls_shape)  # (1, 1, embed_dim) -> (batch, 1, embed_dim)
        x = jnp.concatenate([cls, x], axis=1)  # (batch, 1, embed_dim), (batch, patches, embed_dim) -> (batch, tokens, embed_dim)

        # Add learned positions and run the encoder stack.
        pos_init = nn.initializers.normal(0.02)
        pos_shape = (1, x.shape[1], self.embed_dim)  # -> (1, tokens, embed_dim)
        pos = self.param('pos_embed', pos_init, pos_shape)  # -> (1, tokens, embed_dim)
        x = x + pos  # (batch, tokens, embed_dim)
        for _ in range(self.depth):
            x = EncoderBlock(self.embed_dim, self.num_heads)(x)  # (batch, tokens, embed_dim)

        # Normalize CLS output and project to class logits.
        x = nn.LayerNorm(name='encoder_norm')(x)  # (batch, tokens, embed_dim)
        cls_output = x[:, 0]  # (batch, tokens, embed_dim) -> (batch, embed_dim)
        logits = nn.Dense(self.num_classes, name='head')(cls_output)  # (batch, embed_dim) -> (batch, num_classes)
        return logits  # (batch, num_classes)
