# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
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


# %% [notebook-only]
# Create and run patch embedding: (2, 32, 32, 3) -> (2, 4, 24).
patch_embed = PatchEmbed(embed_dim=24, patch_size=16)
images = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
example_params = patch_embed.init(jax.random.PRNGKey(0), images)
patch_tokens = patch_embed.apply(example_params, images)  # (2, 32, 32, 3) -> (2, 4, 24)
print("patch_tokens shape:", patch_tokens.shape)

# %%
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


# %% [notebook-only]
# Create and run vision self-attention: (2, 5, 24) -> (2, 5, 24).
example_attention = MultiHeadSelfAttention(embed_dim=24, num_heads=4)
tokens = jnp.ones((2, 5, 24))  # -> (2, 5, 24)
example_params = example_attention.init(jax.random.PRNGKey(1), tokens)
example_attended = example_attention.apply(example_params, tokens)  # (2, 5, 24) -> (2, 5, 24)
print("attended shape:", example_attended.shape)

# %%
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


# %% [notebook-only]
# Create and run one ViT encoder block: (2, 5, 24) -> (2, 5, 24).
example_block = EncoderBlock(embed_dim=24, num_heads=4, mlp_dim=48)
tokens = jnp.ones((2, 5, 24))  # -> (2, 5, 24)
example_params = example_block.init(jax.random.PRNGKey(2), tokens)
encoded_tokens = example_block.apply(example_params, tokens)  # (2, 5, 24) -> (2, 5, 24)
print("encoded_tokens shape:", encoded_tokens.shape)

# %%
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


# %% [notebook-only]
# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
example_model = VisionTransformer(num_classes=1000)
example_test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input)
example_logits = example_model.apply(example_params, example_test_input)  # (2, 224, 224, 3) -> (2, 1000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny synthetic image batch.
model = VisionTransformer(num_classes=2, embed_dim=48, depth=1, num_heads=4)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # (2, 224, 224, 3) -> (2, 2)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2) -> (2, 2)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 2)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 2), (2, 2) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
