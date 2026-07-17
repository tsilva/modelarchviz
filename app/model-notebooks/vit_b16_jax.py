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
        # @arch patchembed.__call__.projection-nn-conv-self-embed_dim-self-patch_size-self-patch_size-stride:start
        projection = nn.Conv(self.embed_dim, (self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), name='proj')
        # @arch patchembed.__call__.projection-nn-conv-self-embed_dim-self-patch_size-self-patch_size-stride:end
        # @arch patchembed.__call__.x-projection-x:start
        x = projection(x)  # (batch, height, width, channels) -> (batch, grid, grid, embed_dim)
        # @arch patchembed.__call__.x-projection-x:end

        # Flatten patches into a token sequence: (batch, grid, grid, embed_dim) -> (batch, patches, embed_dim).
        # @arch patchembed.__call__.batch_size-x-shape-n:start
        batch_size = x.shape[0]  # (batch, grid, grid, embed_dim) -> scalar
        # @arch patchembed.__call__.batch_size-x-shape-n:end
        # @arch patchembed.__call__.sequence_shape-batch_size-n-self-embed_dim:start
        sequence_shape = (batch_size, -1, self.embed_dim)  # -> (batch, patches, embed_dim)
        # @arch patchembed.__call__.sequence_shape-batch_size-n-self-embed_dim:end
        # @arch patchembed.__call__.x-x-reshape-sequence_shape:start
        x = x.reshape(sequence_shape)  # (batch, grid, grid, embed_dim) -> (batch, patches, embed_dim)
        # @arch patchembed.__call__.x-x-reshape-sequence_shape:end
        # @arch patchembed.__call__.return-x:start
        return x  # (batch, patches, embed_dim)
        # @arch patchembed.__call__.return-x:end


# %% [notebook-only]
# Create and run patch embedding: (2, 32, 32, 3) -> (2, 4, 24).
patch_embed = PatchEmbed(embed_dim=24, patch_size=16)
images = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
example_params = patch_embed.init(jax.random.PRNGKey(0), images)
patch_tokens = patch_embed.apply(example_params, images)  # (2, 32, 32, 3) -> (2, 4, 24)
print("patch_tokens shape:", patch_tokens.shape)

# %%
# @arch class-multiheadselfattention-nn-module:start
class MultiHeadSelfAttention(nn.Module):
# @arch class-multiheadselfattention-nn-module:end
    # @arch multiheadselfattention.embed_dim-int-n:start
    embed_dim: int = 768
    # @arch multiheadselfattention.embed_dim-int-n:end
    # @arch multiheadselfattention.num_heads-int-n:start
    num_heads: int = 12
    # @arch multiheadselfattention.num_heads-int-n:end

    # @arch multiheadselfattention.nn-compact:start
    @nn.compact
    # @arch multiheadselfattention.nn-compact:end
    # @arch multiheadselfattention.def-__call__-self-x:start
    def __call__(self, x):
    # @arch multiheadselfattention.def-__call__-self-x:end
        # Project tokens into per-head query, key, and value tensors.
        # @arch multiheadselfattention.__call__.batch_size-x-shape-n:start
        batch_size = x.shape[0]  # (batch, tokens, embed_dim) -> scalar
        # @arch multiheadselfattention.__call__.batch_size-x-shape-n:end
        # @arch multiheadselfattention.__call__.tokens-x-shape-n:start
        tokens = x.shape[1]  # (batch, tokens, embed_dim) -> scalar
        # @arch multiheadselfattention.__call__.tokens-x-shape-n:end
        # @arch multiheadselfattention.__call__.head_dim-self-embed_dim-self-num_heads:start
        head_dim = self.embed_dim // self.num_heads  # scalar
        # @arch multiheadselfattention.__call__.head_dim-self-embed_dim-self-num_heads:end
        # @arch multiheadselfattention.__call__.q-nn-dense-self-embed_dim-name-q_proj-x:start
        q = nn.Dense(self.embed_dim, name='q_proj')(x)  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.__call__.q-nn-dense-self-embed_dim-name-q_proj-x:end
        # @arch multiheadselfattention.__call__.k-nn-dense-self-embed_dim-name-k_proj-x:start
        k = nn.Dense(self.embed_dim, name='k_proj')(x)  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.__call__.k-nn-dense-self-embed_dim-name-k_proj-x:end
        # @arch multiheadselfattention.__call__.v-nn-dense-self-embed_dim-name-v_proj-x:start
        v = nn.Dense(self.embed_dim, name='v_proj')(x)  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.__call__.v-nn-dense-self-embed_dim-name-v_proj-x:end

        # Split model width across heads: (batch, tokens, embed_dim) -> (batch, heads, tokens, head_dim).
        # @arch multiheadselfattention.__call__.head_shape-batch_size-tokens-self-num_heads-head_dim:start
        head_shape = (batch_size, tokens, self.num_heads, head_dim)
        # @arch multiheadselfattention.__call__.head_shape-batch_size-tokens-self-num_heads-head_dim:end
        # @arch multiheadselfattention.__call__.q-q-reshape-head_shape:start
        q = q.reshape(head_shape)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.__call__.q-q-reshape-head_shape:end
        # @arch multiheadselfattention.__call__.q-jnp-transpose-q-n-n-n-n:start
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        # @arch multiheadselfattention.__call__.q-jnp-transpose-q-n-n-n-n:end
        # @arch multiheadselfattention.__call__.k-k-reshape-head_shape:start
        k = k.reshape(head_shape)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.__call__.k-k-reshape-head_shape:end
        # @arch multiheadselfattention.__call__.k-jnp-transpose-k-n-n-n-n:start
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        # @arch multiheadselfattention.__call__.k-jnp-transpose-k-n-n-n-n:end
        # @arch multiheadselfattention.__call__.v-v-reshape-head_shape:start
        v = v.reshape(head_shape)  # (batch, tokens, embed_dim) -> (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.__call__.v-v-reshape-head_shape:end
        # @arch multiheadselfattention.__call__.v-jnp-transpose-v-n-n-n-n:start
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        # @arch multiheadselfattention.__call__.v-jnp-transpose-v-n-n-n-n:end

        # Compute scaled dot-product attention over all image tokens.
        # @arch multiheadselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n:start
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, tokens, head_dim) -> (batch, heads, head_dim, tokens)
        # @arch multiheadselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n:end
        # @arch multiheadselfattention.__call__.scores-q-key_transpose:start
        scores = q @ key_transpose  # (batch, heads, tokens, head_dim), (batch, heads, head_dim, tokens) -> (batch, heads, tokens, tokens)
        # @arch multiheadselfattention.__call__.scores-q-key_transpose:end
        # @arch multiheadselfattention.__call__.scale-head_dim-n:start
        scale = head_dim ** -0.5  # scalar
        # @arch multiheadselfattention.__call__.scale-head_dim-n:end
        # @arch multiheadselfattention.__call__.attn_scores-scores-scale:start
        attn_scores = scores * scale  # (batch, heads, tokens, tokens)
        # @arch multiheadselfattention.__call__.attn_scores-scores-scale:end
        # @arch multiheadselfattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n:start
        attn_weights = nn.softmax(attn_scores, axis=-1)  # (batch, heads, tokens, tokens)
        # @arch multiheadselfattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n:end

        # Mix values, merge heads, and project back to embedding width.
        # @arch multiheadselfattention.__call__.context-attn_weights-v:start
        context = attn_weights @ v  # (batch, heads, tokens, tokens), (batch, heads, tokens, head_dim) -> (batch, heads, tokens, head_dim)
        # @arch multiheadselfattention.__call__.context-attn_weights-v:end
        # @arch multiheadselfattention.__call__.context-jnp-transpose-context-n-n-n-n:start
        context = jnp.transpose(context, (0, 2, 1, 3))  # (batch, heads, tokens, head_dim) -> (batch, tokens, heads, head_dim)
        # @arch multiheadselfattention.__call__.context-jnp-transpose-context-n-n-n-n:end
        # @arch multiheadselfattention.__call__.merged_shape-batch_size-tokens-self-embed_dim:start
        merged_shape = (batch_size, tokens, self.embed_dim)
        # @arch multiheadselfattention.__call__.merged_shape-batch_size-tokens-self-embed_dim:end
        # @arch multiheadselfattention.__call__.merged-context-reshape-merged_shape:start
        merged = context.reshape(merged_shape)  # (batch, tokens, heads, head_dim) -> (batch, tokens, embed_dim)
        # @arch multiheadselfattention.__call__.merged-context-reshape-merged_shape:end
        # @arch multiheadselfattention.__call__.out-nn-dense-self-embed_dim-name-out_proj-merged:start
        out = nn.Dense(self.embed_dim, name='out_proj')(merged)  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.__call__.out-nn-dense-self-embed_dim-name-out_proj-merged:end
        # @arch multiheadselfattention.__call__.return-out:start
        return out  # (batch, tokens, embed_dim)
        # @arch multiheadselfattention.__call__.return-out:end


# %% [notebook-only]
# Create and run vision self-attention: (2, 5, 24) -> (2, 5, 24).
example_attention = MultiHeadSelfAttention(embed_dim=24, num_heads=4)
tokens = jnp.ones((2, 5, 24))  # -> (2, 5, 24)
example_params = example_attention.init(jax.random.PRNGKey(1), tokens)
example_attended = example_attention.apply(example_params, tokens)  # (2, 5, 24) -> (2, 5, 24)
print("attended shape:", example_attended.shape)

# %%
# @arch class-encoderblock-nn-module:start
class EncoderBlock(nn.Module):
# @arch class-encoderblock-nn-module:end
    # @arch encoderblock.embed_dim-int-n:start
    embed_dim: int = 768
    # @arch encoderblock.embed_dim-int-n:end
    # @arch encoderblock.num_heads-int-n:start
    num_heads: int = 12
    # @arch encoderblock.num_heads-int-n:end
    # @arch encoderblock.mlp_dim-int-n:start
    mlp_dim: int = 3072
    # @arch encoderblock.mlp_dim-int-n:end

    # @arch encoderblock.nn-compact:start
    @nn.compact
    # @arch encoderblock.nn-compact:end
    # @arch encoderblock.def-__call__-self-x:start
    def __call__(self, x):
    # @arch encoderblock.def-__call__-self-x:end
        # Apply self-attention with a residual connection.
        # @arch encoderblock.__call__.y-nn-layernorm-name-ln_n-x:start
        y = nn.LayerNorm(name='ln_1')(x)  # (batch, tokens, embed_dim)
        # @arch encoderblock.__call__.y-nn-layernorm-name-ln_n-x:end
        # @arch encoderblock.__call__.y-multiheadselfattention-self-embed_dim-self-num_heads-y:start
        y = MultiHeadSelfAttention(self.embed_dim, self.num_heads)(y)  # (batch, tokens, embed_dim)
        # @arch encoderblock.__call__.y-multiheadselfattention-self-embed_dim-self-num_heads-y:end
        # @arch encoderblock.__call__.x-x-y:start
        x = x + y  # (batch, tokens, embed_dim)
        # @arch encoderblock.__call__.x-x-y:end

        # Apply MLP with a residual connection.
        # @arch encoderblock.__call__.y-nn-layernorm-name-ln_n-x.2:start
        y = nn.LayerNorm(name='ln_2')(x)  # (batch, tokens, embed_dim)
        # @arch encoderblock.__call__.y-nn-layernorm-name-ln_n-x.2:end
        # @arch encoderblock.__call__.y-nn-dense-self-mlp_dim-name-mlp_fcn-y:start
        y = nn.Dense(self.mlp_dim, name='mlp_fc1')(y)  # (batch, tokens, embed_dim) -> (batch, tokens, mlp_dim)
        # @arch encoderblock.__call__.y-nn-dense-self-mlp_dim-name-mlp_fcn-y:end
        # @arch encoderblock.__call__.y-nn-gelu-y:start
        y = nn.gelu(y)  # (batch, tokens, mlp_dim)
        # @arch encoderblock.__call__.y-nn-gelu-y:end
        # @arch encoderblock.__call__.y-nn-dense-self-embed_dim-name-mlp_fcn-y:start
        y = nn.Dense(self.embed_dim, name='mlp_fc2')(y)  # (batch, tokens, mlp_dim) -> (batch, tokens, embed_dim)
        # @arch encoderblock.__call__.y-nn-dense-self-embed_dim-name-mlp_fcn-y:end
        # @arch encoderblock.__call__.out-x-y:start
        out = x + y  # (batch, tokens, embed_dim)
        # @arch encoderblock.__call__.out-x-y:end
        # @arch encoderblock.__call__.return-out:start
        return out  # (batch, tokens, embed_dim)
        # @arch encoderblock.__call__.return-out:end


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
        # @arch visiontransformer.__call__.x-patchembed-self-embed_dim-x:start
        x = PatchEmbed(self.embed_dim)(x)  # (batch, height, width, channels) -> (batch, patches, embed_dim)
        # @arch visiontransformer.__call__.x-patchembed-self-embed_dim-x:end
        # @arch visiontransformer.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-embed_dim:start
        cls = self.param('cls_token', nn.initializers.zeros, (1, 1, self.embed_dim))  # -> (1, 1, embed_dim)
        # @arch visiontransformer.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-embed_dim:end
        # @arch visiontransformer.__call__.batch_size-x-shape-n:start
        batch_size = x.shape[0]  # (batch, patches, embed_dim) -> scalar
        # @arch visiontransformer.__call__.batch_size-x-shape-n:end
        # @arch visiontransformer.__call__.cls_shape-batch_size-n-n:start
        cls_shape = (batch_size, 1, 1)  # -> (batch, 1, 1)
        # @arch visiontransformer.__call__.cls_shape-batch_size-n-n:end
        # @arch visiontransformer.__call__.cls-jnp-tile-cls-cls_shape:start
        cls = jnp.tile(cls, cls_shape)  # (1, 1, embed_dim) -> (batch, 1, embed_dim)
        # @arch visiontransformer.__call__.cls-jnp-tile-cls-cls_shape:end
        # @arch visiontransformer.__call__.x-jnp-concatenate-cls-x-axis-n:start
        x = jnp.concatenate([cls, x], axis=1)  # (batch, 1, embed_dim), (batch, patches, embed_dim) -> (batch, tokens, embed_dim)
        # @arch visiontransformer.__call__.x-jnp-concatenate-cls-x-axis-n:end

        # Add learned positions and run the encoder stack.
        # @arch visiontransformer.__call__.pos_init-nn-initializers-normal-n:start
        pos_init = nn.initializers.normal(0.02)
        # @arch visiontransformer.__call__.pos_init-nn-initializers-normal-n:end
        # @arch visiontransformer.__call__.pos_shape-n-x-shape-n-self-embed_dim:start
        pos_shape = (1, x.shape[1], self.embed_dim)  # -> (1, tokens, embed_dim)
        # @arch visiontransformer.__call__.pos_shape-n-x-shape-n-self-embed_dim:end
        # @arch visiontransformer.__call__.pos-self-param-pos_embed-pos_init-pos_shape:start
        pos = self.param('pos_embed', pos_init, pos_shape)  # -> (1, tokens, embed_dim)
        # @arch visiontransformer.__call__.pos-self-param-pos_embed-pos_init-pos_shape:end
        # @arch visiontransformer.__call__.x-x-pos:start
        x = x + pos  # (batch, tokens, embed_dim)
        # @arch visiontransformer.__call__.x-x-pos:end
        # @arch visiontransformer.__call__.for-_-in-range-self-depth:start
        for _ in range(self.depth):
        # @arch visiontransformer.__call__.for-_-in-range-self-depth:end
            # @arch visiontransformer.__call__.x-encoderblock-self-embed_dim-self-num_heads-x:start
            x = EncoderBlock(self.embed_dim, self.num_heads)(x)  # (batch, tokens, embed_dim)
            # @arch visiontransformer.__call__.x-encoderblock-self-embed_dim-self-num_heads-x:end

        # Normalize CLS output and project to class logits.
        # @arch visiontransformer.__call__.x-nn-layernorm-name-encoder_norm-x:start
        x = nn.LayerNorm(name='encoder_norm')(x)  # (batch, tokens, embed_dim)
        # @arch visiontransformer.__call__.x-nn-layernorm-name-encoder_norm-x:end
        # @arch visiontransformer.__call__.cls_output-x-n:start
        cls_output = x[:, 0]  # (batch, tokens, embed_dim) -> (batch, embed_dim)
        # @arch visiontransformer.__call__.cls_output-x-n:end
        # @arch visiontransformer.__call__.logits-nn-dense-self-num_classes-name-head-cls_output:start
        logits = nn.Dense(self.num_classes, name='head')(cls_output)  # (batch, embed_dim) -> (batch, num_classes)
        # @arch visiontransformer.__call__.logits-nn-dense-self-num_classes-name-head-cls_output:end
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
