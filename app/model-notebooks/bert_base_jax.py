# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-bertembeddings-nn-module:start
class BertEmbeddings(nn.Module):
# @arch class-bertembeddings-nn-module:end
    vocab_size: int = 30522
    hidden_size: int = 768
    max_position: int = 512
    type_vocab_size: int = 2

    # @arch bertembeddings.nn-compact:start
    @nn.compact
    # @arch bertembeddings.nn-compact:end
    # @arch bertembeddings.def-__call__-self-input_ids-token_type_ids-train-false:start
    def __call__(self, input_ids, token_type_ids, train=False):
    # @arch bertembeddings.def-__call__-self-input_ids-token_type_ids-train-false:end
        # Combine token, position, and segment embeddings: (batch, steps) -> (batch, steps, hidden_size).
        # @arch bertembeddings.__call__.positions-jnp-arange-input_ids-shape-n:start
        positions = jnp.arange(input_ids.shape[1])  # -> (steps)
        # @arch bertembeddings.__call__.positions-jnp-arange-input_ids-shape-n:end
        # @arch bertembeddings.__call__.x-nn-embed-self-vocab_size-self-hidden_size-name-word_embeddings-input_i:start
        x = nn.Embed(self.vocab_size, self.hidden_size, name='word_embeddings')(input_ids)  # (batch, steps) -> (batch, steps, hidden_size)
        # @arch bertembeddings.__call__.x-nn-embed-self-vocab_size-self-hidden_size-name-word_embeddings-input_i:end
        # @arch bertembeddings.__call__.position_embeddings-nn-embed-self-max_position-self-hidden_size-name-pos:start
        position_embeddings = nn.Embed(self.max_position, self.hidden_size, name='position_embeddings')(positions)  # (steps) -> (steps, hidden_size)
        # @arch bertembeddings.__call__.position_embeddings-nn-embed-self-max_position-self-hidden_size-name-pos:end
        # @arch bertembeddings.__call__.position_embeddings-position_embeddings-none:start
        position_embeddings = position_embeddings[None, :, :]  # (steps, hidden_size) -> (1, steps, hidden_size)
        # @arch bertembeddings.__call__.position_embeddings-position_embeddings-none:end
        # @arch bertembeddings.__call__.x-x-position_embeddings:start
        x = x + position_embeddings  # (batch, steps, hidden_size)
        # @arch bertembeddings.__call__.x-x-position_embeddings:end
        # @arch bertembeddings.__call__.token_type_embeddings-nn-embed-self-type_vocab_size-self-hidden_size-nam:start
        token_type_embeddings = nn.Embed(self.type_vocab_size, self.hidden_size, name='token_type_embeddings')(token_type_ids)  # (batch, steps) -> (batch, steps, hidden_size)
        # @arch bertembeddings.__call__.token_type_embeddings-nn-embed-self-type_vocab_size-self-hidden_size-nam:end
        # @arch bertembeddings.__call__.x-x-token_type_embeddings:start
        x = x + token_type_embeddings  # (batch, steps, hidden_size)
        # @arch bertembeddings.__call__.x-x-token_type_embeddings:end

        # Normalize and regularize embeddings while preserving shape.
        # @arch bertembeddings.__call__.x-nn-layernorm-name-layernorm-x:start
        x = nn.LayerNorm(name='LayerNorm')(x)  # (batch, steps, hidden_size)
        # @arch bertembeddings.__call__.x-nn-layernorm-name-layernorm-x:end
        # @arch bertembeddings.__call__.x-nn-dropout-n-deterministic-not-train-x:start
        x = nn.Dropout(0.1, deterministic=not train)(x)  # (batch, steps, hidden_size)
        # @arch bertembeddings.__call__.x-nn-dropout-n-deterministic-not-train-x:end
        # @arch bertembeddings.__call__.return-x:start
        return x  # (batch, steps, hidden_size)
        # @arch bertembeddings.__call__.return-x:end


# %% [notebook-only]
# Create and run the embedding block: (2, 4) -> (2, 4, 12).
embeddings = BertEmbeddings(vocab_size=20, hidden_size=12, max_position=8)
example_input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
example_token_type_ids = jnp.zeros((2, 4), dtype=jnp.int32)  # -> (2, 4)
example_params = embeddings.init(jax.random.PRNGKey(0), example_input_ids, example_token_type_ids, train=False)
example_embedded = embeddings.apply(example_params, example_input_ids, example_token_type_ids, train=False)  # (2, 4), (2, 4) -> (2, 4, 12)
print("embedded shape:", example_embedded.shape)

# %%
# @arch class-bertselfattention-nn-module:start
class BertSelfAttention(nn.Module):
# @arch class-bertselfattention-nn-module:end
    # @arch bertselfattention.hidden_size-int-n:start
    hidden_size: int = 768
    # @arch bertselfattention.hidden_size-int-n:end
    # @arch bertselfattention.num_heads-int-n:start
    num_heads: int = 12
    # @arch bertselfattention.num_heads-int-n:end

    # @arch bertselfattention.nn-compact:start
    @nn.compact
    # @arch bertselfattention.nn-compact:end
    # @arch bertselfattention.def-__call__-self-x-attention_mask-none:start
    def __call__(self, x, attention_mask=None):
    # @arch bertselfattention.def-__call__-self-x-attention_mask-none:end
        # Project token states into per-head query, key, and value tensors.
        # @arch bertselfattention.__call__.batch_size-x-shape-n:start
        batch_size = x.shape[0]  # (batch, steps, hidden_size) -> scalar
        # @arch bertselfattention.__call__.batch_size-x-shape-n:end
        # @arch bertselfattention.__call__.steps-x-shape-n:start
        steps = x.shape[1]  # (batch, steps, hidden_size) -> scalar
        # @arch bertselfattention.__call__.steps-x-shape-n:end
        # @arch bertselfattention.__call__.head_dim-self-hidden_size-self-num_heads:start
        head_dim = self.hidden_size // self.num_heads  # scalar
        # @arch bertselfattention.__call__.head_dim-self-hidden_size-self-num_heads:end
        # @arch bertselfattention.__call__.q-nn-dense-self-hidden_size-name-q_proj-x:start
        q = nn.Dense(self.hidden_size, name='q_proj')(x)  # (batch, steps, hidden_size)
        # @arch bertselfattention.__call__.q-nn-dense-self-hidden_size-name-q_proj-x:end
        # @arch bertselfattention.__call__.k-nn-dense-self-hidden_size-name-k_proj-x:start
        k = nn.Dense(self.hidden_size, name='k_proj')(x)  # (batch, steps, hidden_size)
        # @arch bertselfattention.__call__.k-nn-dense-self-hidden_size-name-k_proj-x:end
        # @arch bertselfattention.__call__.v-nn-dense-self-hidden_size-name-v_proj-x:start
        v = nn.Dense(self.hidden_size, name='v_proj')(x)  # (batch, steps, hidden_size)
        # @arch bertselfattention.__call__.v-nn-dense-self-hidden_size-name-v_proj-x:end

        # Split model width across heads: (batch, steps, hidden_size) -> (batch, heads, steps, head_dim).
        # @arch bertselfattention.__call__.head_shape-batch_size-steps-self-num_heads-head_dim:start
        head_shape = (batch_size, steps, self.num_heads, head_dim)
        # @arch bertselfattention.__call__.head_shape-batch_size-steps-self-num_heads-head_dim:end
        # @arch bertselfattention.__call__.q-q-reshape-head_shape:start
        q = q.reshape(head_shape)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.__call__.q-q-reshape-head_shape:end
        # @arch bertselfattention.__call__.q-jnp-transpose-q-n-n-n-n:start
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.__call__.q-jnp-transpose-q-n-n-n-n:end
        # @arch bertselfattention.__call__.k-k-reshape-head_shape:start
        k = k.reshape(head_shape)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.__call__.k-k-reshape-head_shape:end
        # @arch bertselfattention.__call__.k-jnp-transpose-k-n-n-n-n:start
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.__call__.k-jnp-transpose-k-n-n-n-n:end
        # @arch bertselfattention.__call__.v-v-reshape-head_shape:start
        v = v.reshape(head_shape)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.__call__.v-v-reshape-head_shape:end
        # @arch bertselfattention.__call__.v-jnp-transpose-v-n-n-n-n:start
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.__call__.v-jnp-transpose-v-n-n-n-n:end

        # Compute scaled dot-product attention and mask padded keys.
        # @arch bertselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n:start
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        # @arch bertselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n:end
        # @arch bertselfattention.__call__.scores-q-key_transpose:start
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        # @arch bertselfattention.__call__.scores-q-key_transpose:end
        # @arch bertselfattention.__call__.scale-head_dim-n:start
        scale = head_dim ** -0.5  # scalar
        # @arch bertselfattention.__call__.scale-head_dim-n:end
        # @arch bertselfattention.__call__.attn_scores-scores-scale:start
        attn_scores = scores * scale  # (batch, heads, steps, steps)
        # @arch bertselfattention.__call__.attn_scores-scores-scale:end
        # @arch bertselfattention.__call__.if-attention_mask-is-not-none:start
        if attention_mask is not None:
        # @arch bertselfattention.__call__.if-attention_mask-is-not-none:end
            # @arch bertselfattention.__call__.attn_scores-jnp-where-attention_mask-attn_scores-jnp-inf:start
            attn_scores = jnp.where(attention_mask, attn_scores, -jnp.inf)  # (batch, heads, steps, steps)
            # @arch bertselfattention.__call__.attn_scores-jnp-where-attention_mask-attn_scores-jnp-inf:end
        # @arch bertselfattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n:start
        attn_weights = nn.softmax(attn_scores, axis=-1)  # (batch, heads, steps, steps)
        # @arch bertselfattention.__call__.attn_weights-nn-softmax-attn_scores-axis-n:end

        # Mix values, merge heads, and project back to hidden width.
        # @arch bertselfattention.__call__.context-attn_weights-v:start
        context = attn_weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.__call__.context-attn_weights-v:end
        # @arch bertselfattention.__call__.context-jnp-transpose-context-n-n-n-n:start
        context = jnp.transpose(context, (0, 2, 1, 3))  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.__call__.context-jnp-transpose-context-n-n-n-n:end
        # @arch bertselfattention.__call__.merged_shape-batch_size-steps-self-hidden_size:start
        merged_shape = (batch_size, steps, self.hidden_size)
        # @arch bertselfattention.__call__.merged_shape-batch_size-steps-self-hidden_size:end
        # @arch bertselfattention.__call__.merged-context-reshape-merged_shape:start
        merged = context.reshape(merged_shape)  # (batch, steps, heads, head_dim) -> (batch, steps, hidden_size)
        # @arch bertselfattention.__call__.merged-context-reshape-merged_shape:end
        # @arch bertselfattention.__call__.out-nn-dense-self-hidden_size-name-out_proj-merged:start
        out = nn.Dense(self.hidden_size, name='out_proj')(merged)  # (batch, steps, hidden_size)
        # @arch bertselfattention.__call__.out-nn-dense-self-hidden_size-name-out_proj-merged:end
        # @arch bertselfattention.__call__.return-out:start
        return out  # (batch, steps, hidden_size)
        # @arch bertselfattention.__call__.return-out:end


# %% [notebook-only]
# Create and run one BERT self-attention block: (2, 4, 12) -> (2, 4, 12).
example_attention = BertSelfAttention(hidden_size=12, num_heads=3)
example_hidden_states = jnp.ones((2, 4, 12))  # -> (2, 4, 12)
attention_mask = jnp.ones((2, 1, 1, 4), dtype=jnp.bool_)  # -> (2, 1, 1, 4)
example_params = example_attention.init(jax.random.PRNGKey(1), example_hidden_states, attention_mask)
example_attended = example_attention.apply(example_params, example_hidden_states, attention_mask)  # (2, 4, 12), (2, 1, 1, 4) -> (2, 4, 12)
print("attended shape:", example_attended.shape)

# %%
# @arch class-bertlayer-nn-module:start
class BertLayer(nn.Module):
# @arch class-bertlayer-nn-module:end
    # @arch bertlayer.hidden_size-int-n:start
    hidden_size: int = 768
    # @arch bertlayer.hidden_size-int-n:end
    # @arch bertlayer.num_heads-int-n:start
    num_heads: int = 12
    # @arch bertlayer.num_heads-int-n:end
    # @arch bertlayer.intermediate_size-int-n:start
    intermediate_size: int = 3072
    # @arch bertlayer.intermediate_size-int-n:end

    # @arch bertlayer.nn-compact:start
    @nn.compact
    # @arch bertlayer.nn-compact:end
    # @arch bertlayer.def-__call__-self-x-attention_mask-none-train-false:start
    def __call__(self, x, attention_mask=None, train=False):
    # @arch bertlayer.def-__call__-self-x-attention_mask-none-train-false:end
        # Apply self-attention with residual normalization: (batch, steps, hidden_size).
        # @arch bertlayer.__call__.attn-bertselfattention-self-hidden_size-self-num_heads-x-attention_mask:start
        attn = BertSelfAttention(self.hidden_size, self.num_heads)(x, attention_mask)  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.attn-bertselfattention-self-hidden_size-self-num_heads-x-attention_mask:end
        # @arch bertlayer.__call__.attn-nn-dropout-n-deterministic-not-train-attn:start
        attn = nn.Dropout(0.1, deterministic=not train)(attn)  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.attn-nn-dropout-n-deterministic-not-train-attn:end
        # @arch bertlayer.__call__.attn_residual-x-attn:start
        attn_residual = x + attn  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.attn_residual-x-attn:end
        # @arch bertlayer.__call__.x-nn-layernorm-name-attention_norm-attn_residual:start
        x = nn.LayerNorm(name='attention_norm')(attn_residual)  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.x-nn-layernorm-name-attention_norm-attn_residual:end

        # Apply feed-forward block with residual normalization.
        # @arch bertlayer.__call__.ffn-nn-dense-self-intermediate_size-name-intermediate-x:start
        ffn = nn.Dense(self.intermediate_size, name='intermediate')(x)  # (batch, steps, hidden_size) -> (batch, steps, intermediate_size)
        # @arch bertlayer.__call__.ffn-nn-dense-self-intermediate_size-name-intermediate-x:end
        # @arch bertlayer.__call__.ffn-nn-gelu-ffn:start
        ffn = nn.gelu(ffn)  # (batch, steps, intermediate_size)
        # @arch bertlayer.__call__.ffn-nn-gelu-ffn:end
        # @arch bertlayer.__call__.ffn-nn-dense-self-hidden_size-name-output_dense-ffn:start
        ffn = nn.Dense(self.hidden_size, name='output_dense')(ffn)  # (batch, steps, intermediate_size) -> (batch, steps, hidden_size)
        # @arch bertlayer.__call__.ffn-nn-dense-self-hidden_size-name-output_dense-ffn:end
        # @arch bertlayer.__call__.ffn-nn-dropout-n-deterministic-not-train-ffn:start
        ffn = nn.Dropout(0.1, deterministic=not train)(ffn)  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.ffn-nn-dropout-n-deterministic-not-train-ffn:end
        # @arch bertlayer.__call__.ffn_residual-x-ffn:start
        ffn_residual = x + ffn  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.ffn_residual-x-ffn:end
        # @arch bertlayer.__call__.out-nn-layernorm-name-output_norm-ffn_residual:start
        out = nn.LayerNorm(name='output_norm')(ffn_residual)  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.out-nn-layernorm-name-output_norm-ffn_residual:end
        # @arch bertlayer.__call__.return-out:start
        return out  # (batch, steps, hidden_size)
        # @arch bertlayer.__call__.return-out:end


# %% [notebook-only]
# Create and run one encoder layer: (2, 4, 12) -> (2, 4, 12).
layer = BertLayer(hidden_size=12, num_heads=3, intermediate_size=24)
example_hidden_states = jnp.ones((2, 4, 12))  # -> (2, 4, 12)
attention_mask = jnp.ones((2, 1, 1, 4), dtype=jnp.bool_)  # -> (2, 1, 1, 4)
example_params = layer.init(jax.random.PRNGKey(2), example_hidden_states, attention_mask, train=False)
example_layer_output = layer.apply(example_params, example_hidden_states, attention_mask, train=False)  # (2, 4, 12), (2, 1, 1, 4) -> (2, 4, 12)
print("layer_output shape:", example_layer_output.shape)

# %%
class BERTBase(nn.Module):
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12

    @nn.compact
    def __call__(self, input_ids, token_type_ids, attention_mask=None, train=False):
        # Embed tokens and run the encoder stack.
        # @arch bertbase.__call__.x-bertembeddings-self-vocab_size-self-hidden_size-input_ids-token_type_i:start
        x = BertEmbeddings(self.vocab_size, self.hidden_size)(input_ids, token_type_ids, train=train)  # (batch, steps) -> (batch, steps, hidden_size)
        # @arch bertbase.__call__.x-bertembeddings-self-vocab_size-self-hidden_size-input_ids-token_type_i:end
        # @arch bertbase.__call__.for-_-in-range-self-num_layers:start
        for _ in range(self.num_layers):
        # @arch bertbase.__call__.for-_-in-range-self-num_layers:end
            # @arch bertbase.__call__.x-bertlayer-self-hidden_size-x-attention_mask-train-train:start
            x = BertLayer(self.hidden_size)(x, attention_mask, train=train)  # (batch, steps, hidden_size)
            # @arch bertbase.__call__.x-bertlayer-self-hidden_size-x-attention_mask-train-train:end

        # Pool the CLS token and project sequence states to token logits.
        # @arch bertbase.__call__.cls_token-x-n:start
        cls_token = x[:, 0]  # (batch, steps, hidden_size) -> (batch, hidden_size)
        # @arch bertbase.__call__.cls_token-x-n:end
        # @arch bertbase.__call__.pooled_projection-nn-dense-self-hidden_size-name-pooler-cls_token:start
        pooled_projection = nn.Dense(self.hidden_size, name='pooler')(cls_token)  # (batch, hidden_size)
        # @arch bertbase.__call__.pooled_projection-nn-dense-self-hidden_size-name-pooler-cls_token:end
        # @arch bertbase.__call__.pooled-jnp-tanh-pooled_projection:start
        pooled = jnp.tanh(pooled_projection)  # (batch, hidden_size)
        # @arch bertbase.__call__.pooled-jnp-tanh-pooled_projection:end
        # @arch bertbase.__call__.mlm_logits-nn-dense-self-vocab_size-name-mlm_head-x:start
        mlm_logits = nn.Dense(self.vocab_size, name='mlm_head')(x)  # (batch, steps, hidden_size) -> (batch, steps, vocab_size)
        # @arch bertbase.__call__.mlm_logits-nn-dense-self-vocab_size-name-mlm_head-x:end
        outputs = (mlm_logits, pooled)
        return outputs


# %% [notebook-only]
# Create and run a sample token batch.
example_model = BERTBase(vocab_size=30522)
example_input_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)
example_token_type_ids = jnp.zeros((2, 16), dtype=jnp.int32)  # -> (2, 16)
attention_mask = jnp.ones((2, 1, 1, 16), dtype=jnp.bool_)  # -> (2, 1, 1, 16)
example_params = example_model.init(jax.random.PRNGKey(0), example_input_ids, example_token_type_ids, attention_mask)
example_outputs = example_model.apply(example_params, example_input_ids, example_token_type_ids, attention_mask)
mlm_logits = example_outputs[0]  # tuple -> (2, 16, 30522)
pooled = example_outputs[1]  # tuple -> (2, 768)
print("mlm logits shape:", mlm_logits.shape, "pooled shape:", pooled.shape)

# %%
# Train on a tiny masked-token prediction batch.
model = BERTBase(vocab_size=20, hidden_size=12, num_layers=1)
# @arch input_ids-jnp-array-n-n-n-n-n-n-n-n-dtype-jnp-intn:start
input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
# @arch input_ids-jnp-array-n-n-n-n-n-n-n-n-dtype-jnp-intn:end
# @arch token_type_ids-jnp-zeros-n-n-dtype-jnp-intn:start
token_type_ids = jnp.zeros((2, 4), dtype=jnp.int32)  # -> (2, 4)
# @arch token_type_ids-jnp-zeros-n-n-dtype-jnp-intn:end
attention_mask = jnp.ones((2, 1, 1, 4), dtype=jnp.bool_)  # -> (2, 1, 1, 4)
train_targets = jnp.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=jnp.int32)  # -> (2, 4)
# @arch params-model-init-jax-random-prngkey-n-input_ids-token_type_ids-attentio:start
params = model.init(jax.random.PRNGKey(1), input_ids, token_type_ids, attention_mask, train=False)
# @arch params-model-init-jax-random-prngkey-n-input_ids-token_type_ids-attentio:end


def train_step(params, input_ids, token_type_ids, attention_mask, targets, learning_rate=0.1):
    def loss_fn(current_params):
        # @arch train_step.loss_fn.outputs-model-apply-current_params-input_ids-token_type_ids-attention_ma:start
        outputs = model.apply(current_params, input_ids, token_type_ids, attention_mask, train=False)  # (2, 4), (2, 4), (2, 1, 1, 4) -> tuple
        # @arch train_step.loss_fn.outputs-model-apply-current_params-input_ids-token_type_ids-attention_ma:end
        mlm_logits = outputs[0]  # tuple -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, mlm_logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(mlm_logits, axis=-1)  # (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, input_ids, token_type_ids, attention_mask, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
