# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class PositionalEncoding(nn.Module):
    d_model: int = 512

    @nn.compact
    def __call__(self, x):
        # Build sinusoidal position encodings for the current sequence length.
        # @arch positionalencoding.__call__.seq_len-x-shape-n:start
        seq_len = x.shape[1]  # (batch, steps, d_model) -> scalar
        # @arch positionalencoding.__call__.seq_len-x-shape-n:end
        # @arch positionalencoding.__call__.positions-jnp-arange-seq_len:start
        positions = jnp.arange(seq_len)  # -> (steps)
        # @arch positionalencoding.__call__.positions-jnp-arange-seq_len:end
        # @arch positionalencoding.__call__.position-positions-none:start
        position = positions[:, None]  # (steps) -> (steps, 1)
        # @arch positionalencoding.__call__.position-positions-none:end
        # @arch positionalencoding.__call__.even_indices-jnp-arange-n-self-d_model-n:start
        even_indices = jnp.arange(0, self.d_model, 2)  # -> (d_model // 2)
        # @arch positionalencoding.__call__.even_indices-jnp-arange-n-self-d_model-n:end
        # @arch positionalencoding.__call__.scale-jnp-log-n-self-d_model:start
        scale = -jnp.log(10000.0) / self.d_model  # scalar
        # @arch positionalencoding.__call__.scale-jnp-log-n-self-d_model:end
        # @arch positionalencoding.__call__.div_term-jnp-exp-even_indices-scale:start
        div_term = jnp.exp(even_indices * scale)  # (d_model // 2)
        # @arch positionalencoding.__call__.div_term-jnp-exp-even_indices-scale:end
        # @arch positionalencoding.__call__.pe-jnp-zeros-seq_len-self-d_model:start
        pe = jnp.zeros((seq_len, self.d_model))  # -> (steps, d_model)
        # @arch positionalencoding.__call__.pe-jnp-zeros-seq_len-self-d_model:end
        # @arch positionalencoding.__call__.sin_values-jnp-sin-position-div_term:start
        sin_values = jnp.sin(position * div_term)  # (steps, 1), (d_model // 2) -> (steps, d_model // 2)
        # @arch positionalencoding.__call__.sin_values-jnp-sin-position-div_term:end
        # @arch positionalencoding.__call__.cos_values-jnp-cos-position-div_term:start
        cos_values = jnp.cos(position * div_term)  # (steps, 1), (d_model // 2) -> (steps, d_model // 2)
        # @arch positionalencoding.__call__.cos_values-jnp-cos-position-div_term:end
        # @arch positionalencoding.__call__.pe-pe-at-n-n-set-sin_values:start
        pe = pe.at[:, 0::2].set(sin_values)  # (steps, d_model), (steps, d_model // 2) -> (steps, d_model)
        # @arch positionalencoding.__call__.pe-pe-at-n-n-set-sin_values:end
        # @arch positionalencoding.__call__.pe-pe-at-n-n-set-cos_values:start
        pe = pe.at[:, 1::2].set(cos_values)  # (steps, d_model), (steps, d_model // 2) -> (steps, d_model)
        # @arch positionalencoding.__call__.pe-pe-at-n-n-set-cos_values:end
        # @arch positionalencoding.__call__.batch_pe-pe-none:start
        batch_pe = pe[None, :, :]  # (steps, d_model) -> (1, steps, d_model)
        # @arch positionalencoding.__call__.batch_pe-pe-none:end

        # Add position encodings to embeddings: (batch, steps, d_model).
        # @arch positionalencoding.__call__.encoded-x-batch_pe:start
        encoded = x + batch_pe  # (batch, steps, d_model), (1, steps, d_model) -> (batch, steps, d_model)
        # @arch positionalencoding.__call__.encoded-x-batch_pe:end
        # @arch positionalencoding.__call__.return-encoded:start
        return encoded  # (batch, steps, d_model)
        # @arch positionalencoding.__call__.return-encoded:end


# %% [notebook-only]
# Create and run positional encoding: (2, 4, 8) -> (2, 4, 8).
positioner = PositionalEncoding(d_model=8)
embeddings = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
encoded = positioner.init_with_output(jax.random.PRNGKey(0), embeddings)[0]  # (2, 4, 8)
print("encoded shape:", encoded.shape)

# %%
# @arch class-multiheadattention-nn-module:start
class MultiHeadAttention(nn.Module):
# @arch class-multiheadattention-nn-module:end
    # @arch multiheadattention.d_model-int-n:start
    d_model: int = 512
    # @arch multiheadattention.d_model-int-n:end
    # @arch multiheadattention.nhead-int-n:start
    nhead: int = 8
    # @arch multiheadattention.nhead-int-n:end

    # @arch multiheadattention.nn-compact:start
    @nn.compact
    # @arch multiheadattention.nn-compact:end
    # @arch multiheadattention.def-__call__-self-query-key-value-mask-none:start
    def __call__(self, query, key, value, mask=None):
    # @arch multiheadattention.def-__call__-self-query-key-value-mask-none:end
        # Project inputs into per-head query, key, and value tensors.
        # @arch multiheadattention.__call__.batch_size-query-shape-n:start
        batch_size = query.shape[0]  # (batch, query_steps, d_model) -> scalar
        # @arch multiheadattention.__call__.batch_size-query-shape-n:end
        # @arch multiheadattention.__call__.query_steps-query-shape-n:start
        query_steps = query.shape[1]  # (batch, query_steps, d_model) -> scalar
        # @arch multiheadattention.__call__.query_steps-query-shape-n:end
        # @arch multiheadattention.__call__.key_steps-key-shape-n:start
        key_steps = key.shape[1]  # (batch, key_steps, d_model) -> scalar
        # @arch multiheadattention.__call__.key_steps-key-shape-n:end
        # @arch multiheadattention.__call__.head_dim-self-d_model-self-nhead:start
        head_dim = self.d_model // self.nhead  # scalar
        # @arch multiheadattention.__call__.head_dim-self-d_model-self-nhead:end
        # @arch multiheadattention.__call__.q-nn-dense-self-d_model-query:start
        q = nn.Dense(self.d_model)(query)  # (batch, query_steps, d_model)
        # @arch multiheadattention.__call__.q-nn-dense-self-d_model-query:end
        # @arch multiheadattention.__call__.k-nn-dense-self-d_model-key:start
        k = nn.Dense(self.d_model)(key)  # (batch, key_steps, d_model)
        # @arch multiheadattention.__call__.k-nn-dense-self-d_model-key:end
        # @arch multiheadattention.__call__.v-nn-dense-self-d_model-value:start
        v = nn.Dense(self.d_model)(value)  # (batch, key_steps, d_model)
        # @arch multiheadattention.__call__.v-nn-dense-self-d_model-value:end

        # Split model width across heads: (batch, steps, d_model) -> (batch, heads, steps, head_dim).
        # @arch multiheadattention.__call__.q-q-reshape-batch_size-query_steps-self-nhead-head_dim:start
        q = q.reshape(batch_size, query_steps, self.nhead, head_dim)  # (batch, query_steps, d_model) -> (batch, query_steps, heads, head_dim)
        # @arch multiheadattention.__call__.q-q-reshape-batch_size-query_steps-self-nhead-head_dim:end
        # @arch multiheadattention.__call__.q-jnp-transpose-q-n-n-n-n:start
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, query_steps, heads, head_dim) -> (batch, heads, query_steps, head_dim)
        # @arch multiheadattention.__call__.q-jnp-transpose-q-n-n-n-n:end
        # @arch multiheadattention.__call__.k-k-reshape-batch_size-key_steps-self-nhead-head_dim:start
        k = k.reshape(batch_size, key_steps, self.nhead, head_dim)  # (batch, key_steps, d_model) -> (batch, key_steps, heads, head_dim)
        # @arch multiheadattention.__call__.k-k-reshape-batch_size-key_steps-self-nhead-head_dim:end
        # @arch multiheadattention.__call__.k-jnp-transpose-k-n-n-n-n:start
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, key_steps, heads, head_dim) -> (batch, heads, key_steps, head_dim)
        # @arch multiheadattention.__call__.k-jnp-transpose-k-n-n-n-n:end
        # @arch multiheadattention.__call__.v-v-reshape-batch_size-key_steps-self-nhead-head_dim:start
        v = v.reshape(batch_size, key_steps, self.nhead, head_dim)  # (batch, key_steps, d_model) -> (batch, key_steps, heads, head_dim)
        # @arch multiheadattention.__call__.v-v-reshape-batch_size-key_steps-self-nhead-head_dim:end
        # @arch multiheadattention.__call__.v-jnp-transpose-v-n-n-n-n:start
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, key_steps, heads, head_dim) -> (batch, heads, key_steps, head_dim)
        # @arch multiheadattention.__call__.v-jnp-transpose-v-n-n-n-n:end

        # Compute scaled dot-product attention scores.
        # @arch multiheadattention.__call__.key_transpose-jnp-swapaxes-k-n-n:start
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, key_steps, head_dim) -> (batch, heads, head_dim, key_steps)
        # @arch multiheadattention.__call__.key_transpose-jnp-swapaxes-k-n-n:end
        # @arch multiheadattention.__call__.scores-q-key_transpose:start
        scores = q @ key_transpose  # (batch, heads, query_steps, head_dim), (batch, heads, head_dim, key_steps) -> (batch, heads, query_steps, key_steps)
        # @arch multiheadattention.__call__.scores-q-key_transpose:end
        # @arch multiheadattention.__call__.scale-head_dim-n:start
        scale = head_dim ** -0.5  # scalar
        # @arch multiheadattention.__call__.scale-head_dim-n:end
        # @arch multiheadattention.__call__.attn_scores-scores-scale:start
        attn_scores = scores * scale  # (batch, heads, query_steps, key_steps)
        # @arch multiheadattention.__call__.attn_scores-scores-scale:end
        # @arch multiheadattention.__call__.if-mask-is-not-none:start
        if mask is not None:
        # @arch multiheadattention.__call__.if-mask-is-not-none:end
            # @arch multiheadattention.__call__.attn_scores-jnp-where-mask-n-jnp-inf-attn_scores:start
            attn_scores = jnp.where(mask == 0, -jnp.inf, attn_scores)  # (batch, heads, query_steps, key_steps)
            # @arch multiheadattention.__call__.attn_scores-jnp-where-mask-n-jnp-inf-attn_scores:end
        # @arch multiheadattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n:start
        attn_weights = nn.softmax(attn_scores, axis=-1)  # (batch, heads, query_steps, key_steps)
        # @arch multiheadattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n:end

        # Mix values, merge heads, and project back to model width.
        # @arch multiheadattention.__call__.context-attn_weights-v:start
        context = attn_weights @ v  # (batch, heads, query_steps, key_steps), (batch, heads, key_steps, head_dim) -> (batch, heads, query_steps, head_dim)
        # @arch multiheadattention.__call__.context-attn_weights-v:end
        # @arch multiheadattention.__call__.context-jnp-transpose-context-n-n-n-n:start
        context = jnp.transpose(context, (0, 2, 1, 3))  # (batch, heads, query_steps, head_dim) -> (batch, query_steps, heads, head_dim)
        # @arch multiheadattention.__call__.context-jnp-transpose-context-n-n-n-n:end
        # @arch multiheadattention.__call__.merged-context-reshape-batch_size-query_steps-self-d_model:start
        merged = context.reshape(batch_size, query_steps, self.d_model)  # (batch, query_steps, heads, head_dim) -> (batch, query_steps, d_model)
        # @arch multiheadattention.__call__.merged-context-reshape-batch_size-query_steps-self-d_model:end
        # @arch multiheadattention.__call__.out-nn-dense-self-d_model-merged:start
        out = nn.Dense(self.d_model)(merged)  # (batch, query_steps, d_model)
        # @arch multiheadattention.__call__.out-nn-dense-self-d_model-merged:end
        # @arch multiheadattention.__call__.return-out:start
        return out  # (batch, query_steps, d_model)
        # @arch multiheadattention.__call__.return-out:end


# %% [notebook-only]
# Create and run multi-head attention: query and memory -> (2, 3, 8).
example_attention = MultiHeadAttention(d_model=8, nhead=2)
example_query = jnp.ones((2, 3, 8))  # -> (2, 3, 8)
key = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
value = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
example_mask = jnp.ones((1, 1, 3, 4))  # -> (1, 1, 3, 4)
example_params = example_attention.init(jax.random.PRNGKey(1), example_query, key, value, example_mask)
example_attended = example_attention.apply(example_params, example_query, key, value, example_mask)  # inputs -> (2, 3, 8)
print("attended shape:", example_attended.shape)

# %%
# @arch class-encoderlayer-nn-module:start
class EncoderLayer(nn.Module):
# @arch class-encoderlayer-nn-module:end
    # @arch encoderlayer.d_model-int-n:start
    d_model: int = 512
    # @arch encoderlayer.d_model-int-n:end
    # @arch encoderlayer.nhead-int-n:start
    nhead: int = 8
    # @arch encoderlayer.nhead-int-n:end
    # @arch encoderlayer.d_ff-int-n:start
    d_ff: int = 2048
    # @arch encoderlayer.d_ff-int-n:end

    # @arch encoderlayer.nn-compact:start
    @nn.compact
    # @arch encoderlayer.nn-compact:end
    # @arch encoderlayer.def-__call__-self-x:start
    def __call__(self, x):
    # @arch encoderlayer.def-__call__-self-x:end
        # Apply self-attention with residual normalization: (batch, steps, d_model).
        # @arch encoderlayer.__call__.attn-multiheadattention-self-d_model-self-nhead-x-x-x:start
        attn = MultiHeadAttention(self.d_model, self.nhead)(x, x, x)  # (batch, steps, d_model)
        # @arch encoderlayer.__call__.attn-multiheadattention-self-d_model-self-nhead-x-x-x:end
        # @arch encoderlayer.__call__.attn_residual-x-attn:start
        attn_residual = x + attn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        # @arch encoderlayer.__call__.attn_residual-x-attn:end
        # @arch encoderlayer.__call__.x-nn-layernorm-attn_residual:start
        x = nn.LayerNorm()(attn_residual)  # (batch, steps, d_model)
        # @arch encoderlayer.__call__.x-nn-layernorm-attn_residual:end

        # Apply feed-forward block with residual normalization.
        # @arch encoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model:start
        ffn_layers = [nn.Dense(self.d_ff), nn.relu, nn.Dense(self.d_model)]
        # @arch encoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model:end
        # @arch encoderlayer.__call__.ffn-nn-sequential-ffn_layers-x:start
        ffn = nn.Sequential(ffn_layers)(x)  # (batch, steps, d_model)
        # @arch encoderlayer.__call__.ffn-nn-sequential-ffn_layers-x:end
        # @arch encoderlayer.__call__.ffn_residual-x-ffn:start
        ffn_residual = x + ffn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        # @arch encoderlayer.__call__.ffn_residual-x-ffn:end
        # @arch encoderlayer.__call__.out-nn-layernorm-ffn_residual:start
        out = nn.LayerNorm()(ffn_residual)  # (batch, steps, d_model)
        # @arch encoderlayer.__call__.out-nn-layernorm-ffn_residual:end
        # @arch encoderlayer.__call__.return-out:start
        return out  # (batch, steps, d_model)
        # @arch encoderlayer.__call__.return-out:end


# %% [notebook-only]
# Create and run one encoder layer: (2, 4, 8) -> (2, 4, 8).
example_encoder_layer = EncoderLayer(d_model=8, nhead=2, d_ff=16)
encoder_input = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
example_params = example_encoder_layer.init(jax.random.PRNGKey(2), encoder_input)
example_encoder_output = example_encoder_layer.apply(example_params, encoder_input)  # (2, 4, 8) -> (2, 4, 8)
print("encoder_output shape:", example_encoder_output.shape)

# %%
# @arch class-decoderlayer-nn-module:start
class DecoderLayer(nn.Module):
# @arch class-decoderlayer-nn-module:end
    # @arch decoderlayer.d_model-int-n:start
    d_model: int = 512
    # @arch decoderlayer.d_model-int-n:end
    # @arch decoderlayer.nhead-int-n:start
    nhead: int = 8
    # @arch decoderlayer.nhead-int-n:end
    # @arch decoderlayer.d_ff-int-n:start
    d_ff: int = 2048
    # @arch decoderlayer.d_ff-int-n:end

    # @arch decoderlayer.nn-compact:start
    @nn.compact
    # @arch decoderlayer.nn-compact:end
    # @arch decoderlayer.def-__call__-self-x-memory-mask:start
    def __call__(self, x, memory, mask):
    # @arch decoderlayer.def-__call__-self-x-memory-mask:end
        # Apply masked self-attention with residual normalization.
        # @arch decoderlayer.__call__.masked-multiheadattention-self-d_model-self-nhead-x-x-x-mask:start
        masked = MultiHeadAttention(self.d_model, self.nhead)(x, x, x, mask)  # (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.masked-multiheadattention-self-d_model-self-nhead-x-x-x-mask:end
        # @arch decoderlayer.__call__.masked_residual-x-masked:start
        masked_residual = x + masked  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.masked_residual-x-masked:end
        # @arch decoderlayer.__call__.x-nn-layernorm-masked_residual:start
        x = nn.LayerNorm()(masked_residual)  # (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.x-nn-layernorm-masked_residual:end

        # Attend over encoder memory with residual normalization.
        # @arch decoderlayer.__call__.cross-multiheadattention-self-d_model-self-nhead-x-memory-memory:start
        cross = MultiHeadAttention(self.d_model, self.nhead)(x, memory, memory)  # (batch, target_steps, d_model), (batch, source_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.cross-multiheadattention-self-d_model-self-nhead-x-memory-memory:end
        # @arch decoderlayer.__call__.cross_residual-x-cross:start
        cross_residual = x + cross  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.cross_residual-x-cross:end
        # @arch decoderlayer.__call__.x-nn-layernorm-cross_residual:start
        x = nn.LayerNorm()(cross_residual)  # (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.x-nn-layernorm-cross_residual:end

        # Apply feed-forward block with residual normalization.
        # @arch decoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model:start
        ffn_layers = [nn.Dense(self.d_ff), nn.relu, nn.Dense(self.d_model)]
        # @arch decoderlayer.__call__.ffn_layers-nn-dense-self-d_ff-nn-relu-nn-dense-self-d_model:end
        # @arch decoderlayer.__call__.ffn-nn-sequential-ffn_layers-x:start
        ffn = nn.Sequential(ffn_layers)(x)  # (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.ffn-nn-sequential-ffn_layers-x:end
        # @arch decoderlayer.__call__.ffn_residual-x-ffn:start
        ffn_residual = x + ffn  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.ffn_residual-x-ffn:end
        # @arch decoderlayer.__call__.out-nn-layernorm-ffn_residual:start
        out = nn.LayerNorm()(ffn_residual)  # (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.out-nn-layernorm-ffn_residual:end
        # @arch decoderlayer.__call__.return-out:start
        return out  # (batch, target_steps, d_model)
        # @arch decoderlayer.__call__.return-out:end


# %% [notebook-only]
# Create and run one decoder layer: target and memory -> (2, 4, 8).
decoder_layer = DecoderLayer(d_model=8, nhead=2, d_ff=16)
decoder_input = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
encoder_memory = jnp.ones((2, 5, 8))  # -> (2, 5, 8)
example_mask = jnp.ones((1, 1, 4, 4))  # -> (1, 1, 4, 4)
example_params = decoder_layer.init(jax.random.PRNGKey(3), decoder_input, encoder_memory, example_mask)
example_decoder_output = decoder_layer.apply(example_params, decoder_input, encoder_memory, example_mask)  # inputs -> (2, 4, 8)
print("decoder_output shape:", example_decoder_output.shape)

# %%
class Transformer(nn.Module):
    vocab_size: int = 37000
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6

    @nn.compact
    def __call__(self, src_ids, tgt_ids, tgt_mask):
        # Embed and encode the source tokens: (batch, source_steps) -> memory.
        # @arch transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids:start
        src_embedding = nn.Embed(self.vocab_size, self.d_model)(src_ids)  # (batch, source_steps) -> (batch, source_steps, d_model)
        # @arch transformer.__call__.src_embedding-nn-embed-self-vocab_size-self-d_model-src_ids:end
        # @arch transformer.__call__.memory-positionalencoding-self-d_model-src_embedding:start
        memory = PositionalEncoding(self.d_model)(src_embedding)  # (batch, source_steps, d_model)
        # @arch transformer.__call__.memory-positionalencoding-self-d_model-src_embedding:end
        # @arch transformer.__call__.for-_-in-range-self-num_layers:start
        for _ in range(self.num_layers):
        # @arch transformer.__call__.for-_-in-range-self-num_layers:end
            # @arch transformer.__call__.memory-encoderlayer-self-d_model-self-nhead-memory:start
            memory = EncoderLayer(self.d_model, self.nhead)(memory)  # (batch, source_steps, d_model)
            # @arch transformer.__call__.memory-encoderlayer-self-d_model-self-nhead-memory:end

        # Embed target tokens and decode against source memory.
        # @arch transformer.__call__.tgt_embedding-nn-embed-self-vocab_size-self-d_model-tgt_ids:start
        tgt_embedding = nn.Embed(self.vocab_size, self.d_model)(tgt_ids)  # (batch, target_steps) -> (batch, target_steps, d_model)
        # @arch transformer.__call__.tgt_embedding-nn-embed-self-vocab_size-self-d_model-tgt_ids:end
        # @arch transformer.__call__.x-positionalencoding-self-d_model-tgt_embedding:start
        x = PositionalEncoding(self.d_model)(tgt_embedding)  # (batch, target_steps, d_model)
        # @arch transformer.__call__.x-positionalencoding-self-d_model-tgt_embedding:end
        # @arch transformer.__call__.for-_-in-range-self-num_layers.2:start
        for _ in range(self.num_layers):
        # @arch transformer.__call__.for-_-in-range-self-num_layers.2:end
            # @arch transformer.__call__.x-decoderlayer-self-d_model-self-nhead-x-memory-tgt_mask:start
            x = DecoderLayer(self.d_model, self.nhead)(x, memory, tgt_mask)  # (batch, target_steps, d_model)
            # @arch transformer.__call__.x-decoderlayer-self-d_model-self-nhead-x-memory-tgt_mask:end

        # Project decoder states to vocabulary logits.
        # @arch transformer.__call__.logits-nn-dense-self-vocab_size-x:start
        logits = nn.Dense(self.vocab_size)(x)  # (batch, target_steps, d_model) -> (batch, target_steps, vocab_size)
        # @arch transformer.__call__.logits-nn-dense-self-vocab_size-x:end
        return logits  # (batch, target_steps, vocab_size)


# %% [notebook-only]
# Create and run a sample translation batch.
example_model = Transformer(vocab_size=37000)
src_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)
tgt_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)

# Build a causal target mask: (1, 1, 16, 16).
mask_values = jnp.ones((1, 1, 16, 16))  # -> (1, 1, 16, 16)
tgt_mask = jnp.tril(mask_values)  # (1, 1, 16, 16)
example_params = example_model.init(jax.random.PRNGKey(0), src_ids, tgt_ids, tgt_mask)  # inputs -> parameter tree
example_logits = example_model.apply(example_params, src_ids, tgt_ids, tgt_mask)  # (2, 16), (2, 16), (1, 1, 16, 16) -> (2, 16, 37000)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny copy-style token batch.
model = Transformer(vocab_size=20, d_model=16, nhead=4, num_layers=1)
src_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
tgt_ids = jnp.array([[0, 1, 2, 3], [0, 4, 3, 2]], dtype=jnp.int32)  # -> (2, 4)
train_targets = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
mask_values = jnp.ones((1, 1, 4, 4))  # -> (1, 1, 4, 4)
tgt_mask = jnp.tril(mask_values)  # (1, 1, 4, 4)
params = model.init(jax.random.PRNGKey(1), src_ids, tgt_ids, tgt_mask)  # inputs -> parameter tree


def train_step(params, src_ids, tgt_ids, targets, mask, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, src_ids, tgt_ids, mask)  # (2, 4), (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)  # parameter tree -> scalar, gradient tree
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)  # parameter tree
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, src_ids, tgt_ids, train_targets, tgt_mask)  # parameter tree -> parameter tree, scalar

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
