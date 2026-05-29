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
        positions = jnp.arange(input_ids.shape[1])
        x = nn.Embed(self.vocab_size, self.hidden_size, name='word_embeddings')(input_ids)
        position_embeddings = nn.Embed(self.max_position, self.hidden_size, name='position_embeddings')(positions)
        position_embeddings = position_embeddings[None, :, :]
        x = x + position_embeddings
        token_type_embeddings = nn.Embed(self.type_vocab_size, self.hidden_size, name='token_type_embeddings')(token_type_ids)
        x = x + token_type_embeddings

        # Normalize and regularize embeddings while preserving shape.
        x = nn.LayerNorm(name='LayerNorm')(x)
        x = nn.Dropout(0.1, deterministic=not train)(x)
        return x


class BertLayer(nn.Module):
    hidden_size: int = 768
    num_heads: int = 12
    intermediate_size: int = 3072

    @nn.compact
    def __call__(self, x, attention_mask=None, train=False):
        # Apply self-attention with residual normalization: (batch, steps, hidden_size).
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, name='attention')(x, x, mask=attention_mask)
        attn = nn.Dropout(0.1, deterministic=not train)(attn)
        attn_residual = x + attn
        x = nn.LayerNorm(name='attention_norm')(attn_residual)

        # Apply feed-forward block with residual normalization.
        ffn = nn.Dense(self.intermediate_size, name='intermediate')(x)
        ffn = nn.gelu(ffn)
        ffn = nn.Dense(self.hidden_size, name='output_dense')(ffn)
        ffn = nn.Dropout(0.1, deterministic=not train)(ffn)
        ffn_residual = x + ffn
        out = nn.LayerNorm(name='output_norm')(ffn_residual)
        return out


class BERTBase(nn.Module):
    vocab_size: int = 30522
    hidden_size: int = 768
    num_layers: int = 12

    @nn.compact
    def __call__(self, input_ids, token_type_ids, attention_mask=None, train=False):
        # Embed tokens and run the encoder stack.
        x = BertEmbeddings(self.vocab_size, self.hidden_size)(input_ids, token_type_ids, train=train)
        for _ in range(self.num_layers):
            x = BertLayer(self.hidden_size)(x, attention_mask, train=train)

        # Pool the CLS token and project sequence states to token logits.
        cls_token = x[:, 0]
        pooled_projection = nn.Dense(self.hidden_size, name='pooler')(cls_token)
        pooled = jnp.tanh(pooled_projection)
        mlm_logits = nn.Dense(self.vocab_size, name='mlm_head')(x)
        outputs = (mlm_logits, pooled)
        return outputs


# Create and run a sample token batch.
model = BERTBase(vocab_size=30522)
input_ids = jnp.ones((2, 16), dtype=jnp.int32)
token_type_ids = jnp.zeros((2, 16), dtype=jnp.int32)
attention_mask = jnp.ones((2, 1, 1, 16), dtype=jnp.bool_)
params = model.init(jax.random.PRNGKey(0), input_ids, token_type_ids, attention_mask)
outputs = model.apply(params, input_ids, token_type_ids, attention_mask)
mlm_logits = outputs[0]
pooled = outputs[1]

# mlm_logits: (2, 16, 30522), pooled: (2, 768)
