import jax
import jax.numpy as jnp
from flax import linen as nn

class BertEmbeddings(nn.Module):
    vocab_size: int = 30522
    hidden_size: int = 768
    max_position: int = 512
    type_vocab_size: int = 2

    @nn.compact
    def __call__(self, input_ids, token_type_ids, train=False):
        # Combine token, position, and segment embeddings: (batch, steps) -> (batch, steps, hidden_size).
        positions = jnp.arange(input_ids.shape[1])  # -> (steps)
        x = nn.Embed(self.vocab_size, self.hidden_size, name='word_embeddings')(input_ids)  # (batch, steps) -> (batch, steps, hidden_size)
        position_embeddings = nn.Embed(self.max_position, self.hidden_size, name='position_embeddings')(positions)  # (steps) -> (steps, hidden_size)
        position_embeddings = position_embeddings[None, :, :]  # (steps, hidden_size) -> (1, steps, hidden_size)
        x = x + position_embeddings  # (batch, steps, hidden_size)
        token_type_embeddings = nn.Embed(self.type_vocab_size, self.hidden_size, name='token_type_embeddings')(token_type_ids)  # (batch, steps) -> (batch, steps, hidden_size)
        x = x + token_type_embeddings  # (batch, steps, hidden_size)

        # Normalize and regularize embeddings while preserving shape.
        x = nn.LayerNorm(name='LayerNorm')(x)  # (batch, steps, hidden_size)
        x = nn.Dropout(0.1, deterministic=not train)(x)  # (batch, steps, hidden_size)
        return x  # (batch, steps, hidden_size)

class BertSelfAttention(nn.Module):
    hidden_size: int = 768
    num_heads: int = 12

    @nn.compact
    def __call__(self, x, attention_mask=None):
        # Project token states into per-head query, key, and value tensors.
        batch_size = x.shape[0]  # (batch, steps, hidden_size) -> scalar
        steps = x.shape[1]  # (batch, steps, hidden_size) -> scalar
        head_dim = self.hidden_size // self.num_heads  # scalar
        q = nn.Dense(self.hidden_size, name='q_proj')(x)  # (batch, steps, hidden_size)
        k = nn.Dense(self.hidden_size, name='k_proj')(x)  # (batch, steps, hidden_size)
        v = nn.Dense(self.hidden_size, name='v_proj')(x)  # (batch, steps, hidden_size)

        # Split model width across heads: (batch, steps, hidden_size) -> (batch, heads, steps, head_dim).
        head_shape = (batch_size, steps, self.num_heads, head_dim)
        q = q.reshape(head_shape)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        k = k.reshape(head_shape)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3))  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        v = v.reshape(head_shape)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        v = jnp.transpose(v, (0, 2, 1, 3))  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)

        # Compute scaled dot-product attention and mask padded keys.
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        scale = head_dim ** -0.5  # scalar
        attn_scores = scores * scale  # (batch, heads, steps, steps)
        if attention_mask is not None:
            attn_scores = jnp.where(attention_mask, attn_scores, -jnp.inf)  # (batch, heads, steps, steps)
        attn_weights = nn.softmax(attn_scores, axis=-1)  # (batch, heads, steps, steps)

        # Mix values, merge heads, and project back to hidden width.
        context = attn_weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        context = jnp.transpose(context, (0, 2, 1, 3))  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        merged_shape = (batch_size, steps, self.hidden_size)
        merged = context.reshape(merged_shape)  # (batch, steps, heads, head_dim) -> (batch, steps, hidden_size)
        out = nn.Dense(self.hidden_size, name='out_proj')(merged)  # (batch, steps, hidden_size)
        return out  # (batch, steps, hidden_size)

class BertLayer(nn.Module):
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072

    @nn.compact
    def __call__(self, x, attention_mask=None, train=False):
        # Apply self-attention with residual normalization: (batch, steps, hidden_size).
        attn = BertSelfAttention(self.hidden_size, self.num_heads)(x, attention_mask)  # (batch, steps, hidden_size)
        attn = nn.Dropout(0.1, deterministic=not train)(attn)  # (batch, steps, hidden_size)
        attn_residual = x + attn  # (batch, steps, hidden_size)
        x = nn.LayerNorm(name='attention_norm')(attn_residual)  # (batch, steps, hidden_size)

        # Apply feed-forward block with residual normalization.
        ffn = nn.Dense(self.intermediate_size, name='intermediate')(x)  # (batch, steps, hidden_size) -> (batch, steps, intermediate_size)
        ffn = nn.gelu(ffn)  # (batch, steps, intermediate_size)
        ffn = nn.Dense(self.hidden_size, name='output_dense')(ffn)  # (batch, steps, intermediate_size) -> (batch, steps, hidden_size)
        ffn = nn.Dropout(0.1, deterministic=not train)(ffn)  # (batch, steps, hidden_size)
        ffn_residual = x + ffn  # (batch, steps, hidden_size)
        out = nn.LayerNorm(name='output_norm')(ffn_residual)  # (batch, steps, hidden_size)
        return out  # (batch, steps, hidden_size)

class BERTBase(nn.Module):
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12

    @nn.compact
    def __call__(self, input_ids, token_type_ids, attention_mask=None, train=False):
        # Embed tokens and run the encoder stack.
        x = BertEmbeddings(self.vocab_size, self.hidden_size)(input_ids, token_type_ids, train=train)  # (batch, steps) -> (batch, steps, hidden_size)
        for _ in range(self.num_layers):
            x = BertLayer(self.hidden_size)(x, attention_mask, train=train)  # (batch, steps, hidden_size)

        # Pool the CLS token and project sequence states to token logits.
        cls_token = x[:, 0]  # (batch, steps, hidden_size) -> (batch, hidden_size)
        pooled_projection = nn.Dense(self.hidden_size, name='pooler')(cls_token)  # (batch, hidden_size)
        pooled = jnp.tanh(pooled_projection)  # (batch, hidden_size)
        mlm_logits = nn.Dense(self.vocab_size, name='mlm_head')(x)  # (batch, steps, hidden_size) -> (batch, steps, vocab_size)
        outputs = (mlm_logits, pooled)
        return outputs

# Train on a tiny masked-token prediction batch.
model = BERTBase(vocab_size=20, hidden_size=12, num_layers=1)
input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
token_type_ids = jnp.zeros((2, 4), dtype=jnp.int32)  # -> (2, 4)
attention_mask = jnp.ones((2, 1, 1, 4), dtype=jnp.bool_)  # -> (2, 1, 1, 4)
train_targets = jnp.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=jnp.int32)  # -> (2, 4)
params = model.init(jax.random.PRNGKey(1), input_ids, token_type_ids, attention_mask, train=False)


def train_step(params, input_ids, token_type_ids, attention_mask, targets, learning_rate=0.1):
    def loss_fn(current_params):
        outputs = model.apply(current_params, input_ids, token_type_ids, attention_mask, train=False)  # (2, 4), (2, 4), (2, 1, 1, 4) -> tuple
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
