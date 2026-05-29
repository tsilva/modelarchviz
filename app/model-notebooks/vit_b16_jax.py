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


class PatchEmbed(nn.Module):
    embed_dim: int = 768
    patch_size: int = 16

    @nn.compact
    def __call__(self, x):
        # Project image patches: (batch, height, width, channels) -> (batch, grid, grid, embed_dim).
        projection = nn.Conv(self.embed_dim, (self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), name='proj')
        x = projection(x)

        # Flatten patches into a token sequence: (batch, grid, grid, embed_dim) -> (batch, patches, embed_dim).
        batch_size = x.shape[0]
        sequence_shape = (batch_size, -1, self.embed_dim)
        x = x.reshape(sequence_shape)
        return x


class EncoderBlock(nn.Module):
    embed_dim: int = 768
    num_heads: int = 12
    mlp_dim: int = 3072

    @nn.compact
    def __call__(self, x):
        # Apply self-attention with a residual connection.
        y = nn.LayerNorm(name='ln_1')(x)
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, name='attn')(y, y)
        x = x + y

        # Apply MLP with a residual connection.
        y = nn.LayerNorm(name='ln_2')(x)
        y = nn.Dense(self.mlp_dim, name='mlp_fc1')(y)
        y = nn.gelu(y)
        y = nn.Dense(self.embed_dim, name='mlp_fc2')(y)
        out = x + y
        return out


class VisionTransformer(nn.Module):
    num_classes: int = 1000
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12

    @nn.compact
    def __call__(self, x):
        # Convert image patches into tokens and prepend CLS token.
        x = PatchEmbed(self.embed_dim)(x)
        cls = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))
        batch_size = x.shape[0]
        cls_shape = (batch_size, 1, 1)
        cls = jnp.tile(cls, cls_shape)
        x = jnp.concatenate([cls, x], axis=1)

        # Add learned positions and run the encoder stack.
        pos_init = nn.initializers.normal(0.02)
        pos_shape = (1, x.shape[1], self.embed_dim)
        pos = self.param('pos_embed', pos_init, pos_shape)
        x = x + pos
        for _ in range(self.depth):
            x = EncoderBlock(self.embed_dim, self.num_heads)(x)

        # Normalize CLS output and project to class logits.
        x = nn.LayerNorm(name='encoder_norm')(x)
        cls_output = x[:, 0]
        logits = nn.Dense(self.num_classes, name='head')(cls_output)
        return logits


# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
model = VisionTransformer(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)

# logits: (2, 1000)
