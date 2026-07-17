# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class MultiHeadAttention(nn.Module):
    width: int = 512
    num_heads: int = 8

    @nn.compact
    def __call__(self, x, mask=None):
        # Project tokens into query, key, and value tensors.
        batch_size = x.shape[0]  # (batch, tokens, width) -> scalar
        token_count = x.shape[1]  # (batch, tokens, width) -> scalar
        head_dim = self.width // self.num_heads  # scalar
        q = nn.Dense(self.width, name="q_proj")(x)  # (batch, tokens, width)
        k = nn.Dense(self.width, name="k_proj")(x)  # (batch, tokens, width)
        v = nn.Dense(self.width, name="v_proj")(x)  # (batch, tokens, width)

        # Split model width across attention heads.
        head_shape = (batch_size, token_count, self.num_heads, head_dim)
        q = q.reshape(head_shape)  # (batch, tokens, width) -> (batch, tokens, heads, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        k = k.reshape(head_shape)  # (batch, tokens, width) -> (batch, tokens, heads, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)
        v = v.reshape(head_shape)  # (batch, tokens, width) -> (batch, tokens, heads, head_dim)
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, tokens, heads, head_dim) -> (batch, heads, tokens, head_dim)

        # Compute scaled dot-product attention, optionally with causal masking.
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, tokens, head_dim) -> (batch, heads, head_dim, tokens)
        scores = q @ key_transpose  # (batch, heads, tokens, tokens)
        scale = head_dim ** -0.5  # scalar
        scores = scores * scale  # (batch, heads, tokens, tokens)
        if mask is not None:
            scores = jnp.where(mask == 1, scores, -jnp.inf)  # (batch, heads, tokens, tokens)
        weights = nn.softmax(scores, axis=-1)  # (batch, heads, tokens, tokens)

        # Mix values and merge heads back to model width.
        context = weights @ v  # (batch, heads, tokens, head_dim)
        context = jnp.transpose(context, (0, 2, 1, 3))  # (batch, heads, tokens, head_dim) -> (batch, tokens, heads, head_dim)
        merged_shape = (batch_size, token_count, self.width)
        merged = context.reshape(merged_shape)  # (batch, tokens, heads, head_dim) -> (batch, tokens, width)
        out = nn.Dense(self.width, name="out_proj")(merged)  # (batch, tokens, width)
        return out  # (batch, tokens, width)


# %% [notebook-only]
# Create and run attention over four tokens: (2, 4, 32) -> (2, 4, 32).
example_attention = MultiHeadAttention(width=32, num_heads=4)
example_tokens = jnp.ones((2, 4, 32))  # -> (2, 4, 32)
example_mask = jnp.tril(jnp.ones((4, 4))).reshape(1, 1, 4, 4)  # -> (1, 1, 4, 4)
example_params = example_attention.init(jax.random.PRNGKey(0), example_tokens, example_mask)
example_attended = example_attention.apply(example_params, example_tokens, example_mask)  # (2, 4, 32), (1, 1, 4, 4) -> (2, 4, 32)
print("attended shape:", example_attended.shape)

# %%
class TransformerBlock(nn.Module):
    width: int = 512
    num_heads: int = 8
    mlp_ratio: int = 4

    @nn.compact
    def __call__(self, x, mask=None):
        # Apply self-attention with a residual connection.
        attn_input = nn.LayerNorm(name="ln_1")(x)  # (batch, tokens, width)
        attn_output = MultiHeadAttention(self.width, self.num_heads)(attn_input, mask)  # (batch, tokens, width)
        x = x + attn_output  # (batch, tokens, width)

        # Apply MLP with a residual connection.
        mlp_width = self.width * self.mlp_ratio  # scalar
        mlp_input = nn.LayerNorm(name="ln_2")(x)  # (batch, tokens, width)
        mlp_output = nn.Dense(mlp_width, name="mlp_fc1")(mlp_input)  # (batch, tokens, width) -> (batch, tokens, mlp_width)
        mlp_output = nn.gelu(mlp_output)  # (batch, tokens, mlp_width)
        mlp_output = nn.Dense(self.width, name="mlp_fc2")(mlp_output)  # (batch, tokens, mlp_width) -> (batch, tokens, width)
        out = x + mlp_output  # (batch, tokens, width)
        return out  # (batch, tokens, width)


# %% [notebook-only]
# Create and run one transformer block: (2, 4, 32) -> (2, 4, 32).
example_block = TransformerBlock(width=32, num_heads=4)
example_tokens = jnp.ones((2, 4, 32))  # -> (2, 4, 32)
example_params = example_block.init(jax.random.PRNGKey(1), example_tokens)
example_outputs = example_block.apply(example_params, example_tokens)  # (2, 4, 32) -> (2, 4, 32)
print("block output shape:", example_outputs.shape)

# %%
# @arch class-visionencoder-nn-module:start
class VisionEncoder(nn.Module):
# @arch class-visionencoder-nn-module:end
    image_size: int = 224
    patch_size: int = 32
    width: int = 768
    layers: int = 12
    heads: int = 12
    embed_dim: int = 512

    @nn.compact
    def __call__(self, images):
        # Convert images into patch tokens and prepend the CLS token.
        # @arch visionencoder.__call__.projection-nn-conv-self-width-self-patch_size-self-patch_size-strides-se:start
        projection = nn.Conv(self.width, (self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), use_bias=False, name="patch_embed")
        # @arch visionencoder.__call__.projection-nn-conv-self-width-self-patch_size-self-patch_size-strides-se:end
        # @arch visionencoder.__call__.x-projection-images:start
        x = projection(images)  # (batch, height, width, 3) -> (batch, grid, grid, vision_width)
        # @arch visionencoder.__call__.x-projection-images:end
        batch_size = x.shape[0]  # (batch, grid, grid, vision_width) -> scalar
        # @arch visionencoder.__call__.x-x-reshape-batch_size-n-self-width:start
        x = x.reshape((batch_size, -1, self.width))  # (batch, grid, grid, vision_width) -> (batch, patches, vision_width)
        # @arch visionencoder.__call__.x-x-reshape-batch_size-n-self-width:end
        # @arch visionencoder.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-width:start
        cls = self.param("cls_token", nn.initializers.zeros, (1, 1, self.width))  # -> (1, 1, vision_width)
        # @arch visionencoder.__call__.cls-self-param-cls_token-nn-initializers-zeros-n-n-self-width:end
        # @arch visionencoder.__call__.cls-jnp-tile-cls-batch_size-n-n:start
        cls = jnp.tile(cls, (batch_size, 1, 1))  # (1, 1, vision_width) -> (batch, 1, vision_width)
        # @arch visionencoder.__call__.cls-jnp-tile-cls-batch_size-n-n:end
        # @arch visionencoder.__call__.x-jnp-concatenate-cls-x-axis-n:start
        x = jnp.concatenate([cls, x], axis=1)  # (batch, 1, vision_width), (batch, patches, vision_width) -> (batch, tokens, vision_width)
        # @arch visionencoder.__call__.x-jnp-concatenate-cls-x-axis-n:end

        # Add positions and run the visual transformer.
        # @arch visionencoder.__call__.pos_embed-self-param-pos_embed-nn-initializers-zeros-n-x-shape-n-self-wi:start
        pos_embed = self.param("pos_embed", nn.initializers.zeros, (1, x.shape[1], self.width))  # -> (1, tokens, vision_width)
        # @arch visionencoder.__call__.pos_embed-self-param-pos_embed-nn-initializers-zeros-n-x-shape-n-self-wi:end
        # @arch visionencoder.__call__.x-x-pos_embed:start
        x = x + pos_embed  # (batch, tokens, vision_width)
        # @arch visionencoder.__call__.x-x-pos_embed:end
        # @arch visionencoder.__call__.for-_-in-range-self-layers:start
        for _ in range(self.layers):
        # @arch visionencoder.__call__.for-_-in-range-self-layers:end
            # @arch visionencoder.__call__.x-transformerblock-self-width-self-heads-x:start
            x = TransformerBlock(self.width, self.heads)(x)  # (batch, tokens, vision_width)
            # @arch visionencoder.__call__.x-transformerblock-self-width-self-heads-x:end

        # Project the CLS token into the shared embedding space.
        # @arch visionencoder.__call__.x-nn-layernorm-name-ln_post-x:start
        x = nn.LayerNorm(name="ln_post")(x)  # (batch, tokens, vision_width)
        # @arch visionencoder.__call__.x-nn-layernorm-name-ln_post-x:end
        # @arch visionencoder.__call__.cls_output-x-n:start
        cls_output = x[:, 0]  # (batch, tokens, vision_width) -> (batch, vision_width)
        # @arch visionencoder.__call__.cls_output-x-n:end
        # @arch visionencoder.__call__.image_features-nn-dense-self-embed_dim-use_bias-false-name-proj-cls_outp:start
        image_features = nn.Dense(self.embed_dim, use_bias=False, name="proj")(cls_output)  # (batch, vision_width) -> (batch, embed_dim)
        # @arch visionencoder.__call__.image_features-nn-dense-self-embed_dim-use_bias-false-name-proj-cls_outp:end
        return image_features  # (batch, embed_dim)


# %% [notebook-only]
# Create and run a small image encoder: (2, 32, 32, 3) -> (2, 32).
example_encoder = VisionEncoder(image_size=32, patch_size=16, width=32, layers=1, heads=4, embed_dim=32)
example_images = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
example_params = example_encoder.init(jax.random.PRNGKey(2), example_images)
example_features = example_encoder.apply(example_params, example_images)  # (2, 32, 32, 3) -> (2, 32)
print("image features shape:", example_features.shape)

# %%
# @arch class-textencoder-nn-module:start
class TextEncoder(nn.Module):
# @arch class-textencoder-nn-module:end
    vocab_size: int = 49408
    context_length: int = 77
    width: int = 512
    layers: int = 12
    heads: int = 8
    embed_dim: int = 512

    @nn.compact
    def __call__(self, input_ids):
        # Embed text tokens and add learned positions.
        token_count = input_ids.shape[1]  # (batch, tokens) -> scalar
        # @arch textencoder.__call__.token_embeddings-nn-embed-self-vocab_size-self-width-name-token_embeddin:start
        token_embeddings = nn.Embed(self.vocab_size, self.width, name="token_embedding")(input_ids)  # (batch, tokens) -> (batch, tokens, text_width)
        # @arch textencoder.__call__.token_embeddings-nn-embed-self-vocab_size-self-width-name-token_embeddin:end
        # @arch textencoder.__call__.pos_embed-self-param-pos_embed-nn-initializers-zeros-n-self-context_leng:start
        pos_embed = self.param("pos_embed", nn.initializers.zeros, (1, self.context_length, self.width))  # -> (1, context, text_width)
        # @arch textencoder.__call__.pos_embed-self-param-pos_embed-nn-initializers-zeros-n-self-context_leng:end
        # @arch textencoder.__call__.position_embeddings-pos_embed-token_count:start
        position_embeddings = pos_embed[:, :token_count, :]  # (1, context, text_width) -> (1, tokens, text_width)
        # @arch textencoder.__call__.position_embeddings-pos_embed-token_count:end
        # @arch textencoder.__call__.x-token_embeddings-position_embeddings:start
        x = token_embeddings + position_embeddings  # (batch, tokens, text_width)
        # @arch textencoder.__call__.x-token_embeddings-position_embeddings:end

        # Run the text transformer with a causal mask.
        # @arch textencoder.__call__.full_mask-jnp-tril-jnp-ones-self-context_length-self-context_length:start
        full_mask = jnp.tril(jnp.ones((self.context_length, self.context_length)))  # -> (context, context)
        # @arch textencoder.__call__.full_mask-jnp-tril-jnp-ones-self-context_length-self-context_length:end
        # @arch textencoder.__call__.mask-full_mask-token_count-token_count:start
        mask = full_mask[:token_count, :token_count]  # (context, context) -> (tokens, tokens)
        # @arch textencoder.__call__.mask-full_mask-token_count-token_count:end
        # @arch textencoder.__call__.mask-mask-reshape-n-n-token_count-token_count:start
        mask = mask.reshape(1, 1, token_count, token_count)  # (tokens, tokens) -> (1, 1, tokens, tokens)
        # @arch textencoder.__call__.mask-mask-reshape-n-n-token_count-token_count:end
        # @arch textencoder.__call__.for-_-in-range-self-layers:start
        for _ in range(self.layers):
        # @arch textencoder.__call__.for-_-in-range-self-layers:end
            # @arch textencoder.__call__.x-transformerblock-self-width-self-heads-x-mask:start
            x = TransformerBlock(self.width, self.heads)(x, mask)  # (batch, tokens, text_width)
            # @arch textencoder.__call__.x-transformerblock-self-width-self-heads-x-mask:end

        # Pool at the end-of-text token and project to the shared embedding space.
        # @arch textencoder.__call__.x-nn-layernorm-name-ln_final-x:start
        x = nn.LayerNorm(name="ln_final")(x)  # (batch, tokens, text_width)
        # @arch textencoder.__call__.x-nn-layernorm-name-ln_final-x:end
        # @arch textencoder.__call__.eot_indices-jnp-argmax-input_ids-axis-n:start
        eot_indices = jnp.argmax(input_ids, axis=-1)  # (batch, tokens) -> (batch)
        # @arch textencoder.__call__.eot_indices-jnp-argmax-input_ids-axis-n:end
        # @arch textencoder.__call__.batch_indices-jnp-arange-input_ids-shape-n:start
        batch_indices = jnp.arange(input_ids.shape[0])  # -> (batch)
        # @arch textencoder.__call__.batch_indices-jnp-arange-input_ids-shape-n:end
        # @arch textencoder.__call__.pooled-x-batch_indices-eot_indices:start
        pooled = x[batch_indices, eot_indices]  # (batch, tokens, text_width) -> (batch, text_width)
        # @arch textencoder.__call__.pooled-x-batch_indices-eot_indices:end
        # @arch textencoder.__call__.text_features-nn-dense-self-embed_dim-use_bias-false-name-text_projectio:start
        text_features = nn.Dense(self.embed_dim, use_bias=False, name="text_projection")(pooled)  # (batch, text_width) -> (batch, embed_dim)
        # @arch textencoder.__call__.text_features-nn-dense-self-embed_dim-use_bias-false-name-text_projectio:end
        return text_features  # (batch, embed_dim)


# %% [notebook-only]
# Create and run a small text encoder: (2, 8) -> (2, 32).
example_encoder = TextEncoder(vocab_size=100, context_length=8, width=32, layers=1, heads=4, embed_dim=32)
example_input_ids = jnp.array([[1, 5, 10, 99, 0, 0, 0, 0], [1, 8, 11, 12, 99, 0, 0, 0]])  # -> (2, 8)
example_params = example_encoder.init(jax.random.PRNGKey(3), example_input_ids)
example_features = example_encoder.apply(example_params, example_input_ids)  # (2, 8) -> (2, 32)
print("text features shape:", example_features.shape)

# %%
class CLIP(nn.Module):
    embed_dim: int = 512
    image_size: int = 224
    patch_size: int = 32
    vision_width: int = 768
    vision_layers: int = 12
    vision_heads: int = 12
    vocab_size: int = 49408
    context_length: int = 77
    text_width: int = 512
    text_layers: int = 12
    text_heads: int = 8

    @nn.compact
    # @arch clip.def-__call__-self-images-input_ids:start
    def __call__(self, images, input_ids):
    # @arch clip.def-__call__-self-images-input_ids:end
        # Encode each modality into the same embedding width.
        # @arch clip.__call__.image_features-visionencoder-self-image_size-self-patch_size-self-vision:start
        image_features = VisionEncoder(self.image_size, self.patch_size, self.vision_width, self.vision_layers, self.vision_heads, self.embed_dim)(images)  # (batch, height, width, 3) -> (batch, embed_dim)
        # @arch clip.__call__.image_features-visionencoder-self-image_size-self-patch_size-self-vision:end
        # @arch clip.__call__.text_features-textencoder-self-vocab_size-self-context_length-self-text_:start
        text_features = TextEncoder(self.vocab_size, self.context_length, self.text_width, self.text_layers, self.text_heads, self.embed_dim)(input_ids)  # (batch, tokens) -> (batch, embed_dim)
        # @arch clip.__call__.text_features-textencoder-self-vocab_size-self-context_length-self-text_:end

        # Normalize embeddings and score every image against every text.
        # @arch clip.__call__.image_features-image_features-jnp-linalg-norm-image_features-axis-n-keep:start
        image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)  # (batch, embed_dim)
        # @arch clip.__call__.image_features-image_features-jnp-linalg-norm-image_features-axis-n-keep:end
        # @arch clip.__call__.text_features-text_features-jnp-linalg-norm-text_features-axis-n-keepdim:start
        text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)  # (batch, embed_dim)
        # @arch clip.__call__.text_features-text_features-jnp-linalg-norm-text_features-axis-n-keepdim:end
        # @arch clip.__call__.logit_scale-self-param-logit_scale-lambda-key-jnp-log-jnp-array-n-n:start
        logit_scale = self.param("logit_scale", lambda key: jnp.log(jnp.array(1 / 0.07)))  # -> scalar
        # @arch clip.__call__.logit_scale-self-param-logit_scale-lambda-key-jnp-log-jnp-array-n-n:end
        # @arch clip.__call__.logit_scale-jnp-exp-logit_scale:start
        logit_scale = jnp.exp(logit_scale)  # scalar
        # @arch clip.__call__.logit_scale-jnp-exp-logit_scale:end
        # @arch clip.__call__.text_features_t-jnp-swapaxes-text_features-n-n:start
        text_features_t = jnp.swapaxes(text_features, 0, 1)  # (batch, embed_dim) -> (embed_dim, batch)
        # @arch clip.__call__.text_features_t-jnp-swapaxes-text_features-n-n:end
        # @arch clip.__call__.logits_per_image-logit_scale-image_features-text_features_t:start
        logits_per_image = logit_scale * image_features @ text_features_t  # (batch, embed_dim), (embed_dim, batch) -> (batch, batch)
        # @arch clip.__call__.logits_per_image-logit_scale-image_features-text_features_t:end
        # @arch clip.__call__.logits_per_text-jnp-swapaxes-logits_per_image-n-n:start
        logits_per_text = jnp.swapaxes(logits_per_image, 0, 1)  # (batch, batch)
        # @arch clip.__call__.logits_per_text-jnp-swapaxes-logits_per_image-n-n:end
        return logits_per_image, logits_per_text  # two (batch, batch)


# %% [notebook-only]
# Create and run a small CLIP batch.
example_model = CLIP(
    embed_dim=32,
    image_size=32,
    patch_size=16,
    vision_width=32,
    vision_layers=1,
    vision_heads=4,
    vocab_size=100,
    context_length=8,
    text_width=32,
    text_layers=1,
    text_heads=4,
)
example_images = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
example_input_ids = jnp.array([[1, 5, 10, 99, 0, 0, 0, 0], [1, 8, 11, 12, 99, 0, 0, 0]])  # -> (2, 8)
example_params = example_model.init(jax.random.PRNGKey(4), example_images, example_input_ids)
example_logits = example_model.apply(example_params, example_images, example_input_ids)  # (2, 32, 32, 3), (2, 8) -> two (2, 2)
print("image logits shape:", example_logits[0].shape)

# %%
# Train on a tiny aligned image-text batch.
model = CLIP(
    embed_dim=32,
    image_size=32,
    patch_size=16,
    vision_width=32,
    vision_layers=1,
    vision_heads=4,
    vocab_size=100,
    context_length=8,
    text_width=32,
    text_layers=1,
    text_heads=4,
)
images = jnp.zeros((2, 32, 32, 3))  # -> (2, 32, 32, 3)
images = images.at[0, 4:16, 4:16, :].set(1.0)  # (2, 32, 32, 3)
images = images.at[1, 16:28, 16:28, :].set(1.0)  # (2, 32, 32, 3)
input_ids = jnp.array([[1, 5, 10, 99, 0, 0, 0, 0], [1, 8, 11, 12, 99, 0, 0, 0]])  # -> (2, 8)
targets = jnp.arange(images.shape[0])  # -> (2)
params = model.init(jax.random.PRNGKey(5), images, input_ids)


def contrastive_loss(logits_per_image, logits_per_text, targets):
    image_labels = jax.nn.one_hot(targets, logits_per_image.shape[-1])  # (2) -> (2, 2)
    text_labels = jax.nn.one_hot(targets, logits_per_text.shape[-1])  # (2) -> (2, 2)
    image_log_probs = jax.nn.log_softmax(logits_per_image, axis=-1)  # (2, 2)
    text_log_probs = jax.nn.log_softmax(logits_per_text, axis=-1)  # (2, 2)
    image_loss = -jnp.mean(jnp.sum(image_labels * image_log_probs, axis=-1))  # (2, 2), (2, 2) -> scalar
    text_loss = -jnp.mean(jnp.sum(text_labels * text_log_probs, axis=-1))  # (2, 2), (2, 2) -> scalar
    loss = (image_loss + text_loss) / 2  # scalar
    return loss  # scalar


def train_step(params, images, input_ids, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits_per_image, logits_per_text = model.apply(current_params, images, input_ids)  # (2, 32, 32, 3), (2, 8) -> two (2, 2)
        loss = contrastive_loss(logits_per_image, logits_per_text, targets)  # scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit matching image-text pairs with symmetric contrastive loss.
for step in range(3):
    params, loss = train_step(params, images, input_ids, targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
