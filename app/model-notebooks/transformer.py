# %%
import torch
import torch.nn as nn


# %%
class PositionalEncoding(nn.Module):
    # @arch positionalencoding.def-__init__:start
    def __init__(
    # @arch positionalencoding.def-__init__:end
        # @arch positionalencoding.self:start
        self,
        # @arch positionalencoding.self:end
        d_model=512,  # Embedding width.
        max_len=5000  # Maximum supported sequence length.
    ):
        super().__init__()

        # Build sinusoidal position table: (max_len, d_model).
        # @arch positionalencoding.positions-torch-arange-max_len:start
        positions = torch.arange(max_len)  # -> (max_len)
        # @arch positionalencoding.positions-torch-arange-max_len:end
        # @arch positionalencoding.position-positions-unsqueeze-n:start
        position = positions.unsqueeze(1)  # (max_len) -> (max_len, 1)
        # @arch positionalencoding.position-positions-unsqueeze-n:end
        # @arch positionalencoding.even_indices-torch-arange-n-d_model-n:start
        even_indices = torch.arange(0, d_model, 2)  # -> (d_model // 2)
        # @arch positionalencoding.even_indices-torch-arange-n-d_model-n:end
        # @arch positionalencoding.log_base-torch-log-torch-tensor-n:start
        log_base = torch.log(torch.tensor(10000.0))  # -> scalar
        # @arch positionalencoding.log_base-torch-log-torch-tensor-n:end
        # @arch positionalencoding.scale-log_base-d_model:start
        scale = -log_base / d_model  # scalar
        # @arch positionalencoding.scale-log_base-d_model:end
        # @arch positionalencoding.div_term-torch-exp-even_indices-scale:start
        div_term = torch.exp(even_indices * scale)  # (d_model // 2)
        # @arch positionalencoding.div_term-torch-exp-even_indices-scale:end
        # @arch positionalencoding.pe-torch-zeros-max_len-d_model:start
        pe = torch.zeros(max_len, d_model)  # -> (max_len, d_model)
        # @arch positionalencoding.pe-torch-zeros-max_len-d_model:end
        # @arch positionalencoding.sin_values-torch-sin-position-div_term:start
        sin_values = torch.sin(position * div_term)  # (max_len, 1), (d_model // 2) -> (max_len, d_model // 2)
        # @arch positionalencoding.sin_values-torch-sin-position-div_term:end
        # @arch positionalencoding.cos_values-torch-cos-position-div_term:start
        cos_values = torch.cos(position * div_term)  # (max_len, 1), (d_model // 2) -> (max_len, d_model // 2)
        # @arch positionalencoding.cos_values-torch-cos-position-div_term:end
        # @arch positionalencoding.pe-n-n-sin_values:start
        pe[:, 0::2] = sin_values  # (max_len, d_model), (max_len, d_model // 2) -> (max_len, d_model)
        # @arch positionalencoding.pe-n-n-sin_values:end
        # @arch positionalencoding.pe-n-n-cos_values:start
        pe[:, 1::2] = cos_values  # (max_len, d_model), (max_len, d_model // 2) -> (max_len, d_model)
        # @arch positionalencoding.pe-n-n-cos_values:end
        # @arch positionalencoding.self-register_buffer-pe-pe:start
        self.register_buffer('pe', pe)
        # @arch positionalencoding.self-register_buffer-pe-pe:end

    def forward(self, x):
        # Add position encodings to embeddings: (batch, steps, d_model).
        seq_len = x.size(1)  # (batch, steps, d_model) -> scalar
        # @arch positionalencoding.forward.position_encoding-self-pe-seq_len:start
        position_encoding = self.pe[:seq_len]  # (max_len, d_model) -> (steps, d_model)
        # @arch positionalencoding.forward.position_encoding-self-pe-seq_len:end
        # @arch positionalencoding.forward.encoded-x-position_encoding:start
        encoded = x + position_encoding  # (batch, steps, d_model), (steps, d_model) -> (batch, steps, d_model)
        # @arch positionalencoding.forward.encoded-x-position_encoding:end
        # @arch positionalencoding.forward.return-encoded:start
        return encoded  # (batch, steps, d_model)
        # @arch positionalencoding.forward.return-encoded:end


# %% [notebook-only]
position_encoder = PositionalEncoding(d_model=8, max_len=16)
token_embeddings = torch.zeros(2, 4, 8)  # -> (2, 4, 8)
example_position_encoded = position_encoder(token_embeddings)  # (2, 4, 8)
print(example_position_encoded.shape)


# %%
class MultiHeadAttention(nn.Module):
    def __init__(
        # @arch multiheadattention.self:start
        self,
        # @arch multiheadattention.self:end
        d_model=512,  # Embedding width.
        nhead=8  # Number of attention heads.
    ):
        super().__init__()

        # Register explicit Q/K/V projections and the output projection.
        self.nhead = nhead
        # @arch multiheadattention.self-head_dim-d_model-nhead:start
        self.head_dim = d_model // nhead
        # @arch multiheadattention.self-head_dim-d_model-nhead:end
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        # @arch multiheadattention.self-v_proj-nn-linear-d_model-d_model:start
        self.v_proj = nn.Linear(d_model, d_model)
        # @arch multiheadattention.self-v_proj-nn-linear-d_model-d_model:end
        # @arch multiheadattention.self-out_proj-nn-linear-d_model-d_model:start
        self.out_proj = nn.Linear(d_model, d_model)
        # @arch multiheadattention.self-out_proj-nn-linear-d_model-d_model:end

    # @arch multiheadattention.def-forward-self-query-key-value-attn_mask-none:start
    def forward(self, query, key, value, attn_mask=None):
    # @arch multiheadattention.def-forward-self-query-key-value-attn_mask-none:end
        # Project inputs into per-head query, key, and value tensors.
        # @arch multiheadattention.forward.batch_size-query-size-n:start
        batch_size = query.size(0)  # (batch, query_steps, d_model) -> scalar
        # @arch multiheadattention.forward.batch_size-query-size-n:end
        query_steps = query.size(1)  # (batch, query_steps, d_model) -> scalar
        key_steps = key.size(1)  # (batch, key_steps, d_model) -> scalar
        q = self.q_proj(query)  # (batch, query_steps, d_model)
        k = self.k_proj(key)  # (batch, key_steps, d_model)
        # @arch multiheadattention.forward.v-self-v_proj-value:start
        v = self.v_proj(value)  # (batch, key_steps, d_model)
        # @arch multiheadattention.forward.v-self-v_proj-value:end

        # Split model width across heads: (batch, steps, d_model) -> (batch, heads, steps, head_dim).
        q = q.view(batch_size, query_steps, self.nhead, self.head_dim)  # (batch, query_steps, d_model) -> (batch, query_steps, heads, head_dim)
        q = q.transpose(1, 2)  # (batch, query_steps, heads, head_dim) -> (batch, heads, query_steps, head_dim)
        # @arch multiheadattention.forward.k-k-view-batch_size-key_steps-self-nhead-self-head_dim:start
        k = k.view(batch_size, key_steps, self.nhead, self.head_dim)  # (batch, key_steps, d_model) -> (batch, key_steps, heads, head_dim)
        # @arch multiheadattention.forward.k-k-view-batch_size-key_steps-self-nhead-self-head_dim:end
        # @arch multiheadattention.forward.k-k-transpose-n-n:start
        k = k.transpose(1, 2)  # (batch, key_steps, heads, head_dim) -> (batch, heads, key_steps, head_dim)
        # @arch multiheadattention.forward.k-k-transpose-n-n:end
        # @arch multiheadattention.forward.v-v-view-batch_size-key_steps-self-nhead-self-head_dim:start
        v = v.view(batch_size, key_steps, self.nhead, self.head_dim)  # (batch, key_steps, d_model) -> (batch, key_steps, heads, head_dim)
        # @arch multiheadattention.forward.v-v-view-batch_size-key_steps-self-nhead-self-head_dim:end
        # @arch multiheadattention.forward.v-v-transpose-n-n:start
        v = v.transpose(1, 2)  # (batch, key_steps, heads, head_dim) -> (batch, heads, key_steps, head_dim)
        # @arch multiheadattention.forward.v-v-transpose-n-n:end

        # Compute scaled dot-product attention scores.
        key_transpose = k.transpose(-2, -1)  # (batch, heads, key_steps, head_dim) -> (batch, heads, head_dim, key_steps)
        scores = q @ key_transpose  # (batch, heads, query_steps, head_dim), (batch, heads, head_dim, key_steps) -> (batch, heads, query_steps, key_steps)
        # @arch multiheadattention.forward.scale-self-head_dim-n:start
        scale = self.head_dim ** -0.5  # scalar
        # @arch multiheadattention.forward.scale-self-head_dim-n:end
        # @arch multiheadattention.forward.attn_scores-scores-scale:start
        attn_scores = scores * scale  # (batch, heads, query_steps, key_steps)
        # @arch multiheadattention.forward.attn_scores-scores-scale:end
        # @arch multiheadattention.forward.if-attn_mask-is-not-none:start
        if attn_mask is not None:
        # @arch multiheadattention.forward.if-attn_mask-is-not-none:end
            # @arch multiheadattention.forward.mask-attn_mask-none-none:start
            mask = attn_mask[None, None, :, :]  # (query_steps, key_steps) -> (1, 1, query_steps, key_steps)
            # @arch multiheadattention.forward.mask-attn_mask-none-none:end
            attn_scores = attn_scores + mask  # (batch, heads, query_steps, key_steps)
        # @arch multiheadattention.forward.attn_weights-torch-softmax-attn_scores-dim-n:start
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, heads, query_steps, key_steps)
        # @arch multiheadattention.forward.attn_weights-torch-softmax-attn_scores-dim-n:end

        # Mix values, merge heads, and project back to model width.
        context = attn_weights @ v  # (batch, heads, query_steps, key_steps), (batch, heads, key_steps, head_dim) -> (batch, heads, query_steps, head_dim)
        context = context.transpose(1, 2)  # (batch, heads, query_steps, head_dim) -> (batch, query_steps, heads, head_dim)
        # @arch multiheadattention.forward.context-context-contiguous:start
        context = context.contiguous()  # (batch, query_steps, heads, head_dim)
        # @arch multiheadattention.forward.context-context-contiguous:end
        # @arch multiheadattention.forward.merged-context-view-batch_size-query_steps-self-nhead-self-head_dim:start
        merged = context.view(batch_size, query_steps, self.nhead * self.head_dim)  # (batch, query_steps, heads, head_dim) -> (batch, query_steps, d_model)
        # @arch multiheadattention.forward.merged-context-view-batch_size-query_steps-self-nhead-self-head_dim:end
        # @arch multiheadattention.forward.out-self-out_proj-merged:start
        out = self.out_proj(merged)  # (batch, query_steps, d_model)
        # @arch multiheadattention.forward.out-self-out_proj-merged:end
        # @arch multiheadattention.forward.return-out:start
        return out  # (batch, query_steps, d_model)
        # @arch multiheadattention.forward.return-out:end


# %% [notebook-only]
example_attention = MultiHeadAttention(d_model=8, nhead=2)
example_query = torch.randn(2, 3, 8)  # -> (2, 3, 8)
key = torch.randn(2, 4, 8)  # -> (2, 4, 8)
value = torch.randn(2, 4, 8)  # -> (2, 4, 8)
example_attended = example_attention(example_query, key, value)  # (2, 3, 8)
print(example_attended.shape)


# %%
# @arch class-encoderlayer-nn-module:start
class EncoderLayer(nn.Module):
# @arch class-encoderlayer-nn-module:end
    def __init__(
        self,
        # @arch encoderlayer.d_model-n:start
        d_model=512,  # Embedding width.
        # @arch encoderlayer.d_model-n:end
        nhead=8,  # Number of attention heads.
        d_ff=2048  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register self-attention, feed-forward, and normalization layers.
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            # @arch encoderlayer.nn-linear-d_model-d_ff:start
            nn.Linear(d_model, d_ff),
            # @arch encoderlayer.nn-linear-d_model-d_ff:end
            # @arch encoderlayer.nn-relu:start
            nn.ReLU(),
            # @arch encoderlayer.nn-relu:end
            # @arch encoderlayer.nn-linear-d_ff-d_model:start
            nn.Linear(d_ff, d_model),
            # @arch encoderlayer.nn-linear-d_ff-d_model:end
        # @arch encoderlayer.code.4:start
        )
        # @arch encoderlayer.code.4:end
        # @arch encoderlayer.self-normn-nn-layernorm-d_model:start
        self.norm1 = nn.LayerNorm(d_model)
        # @arch encoderlayer.self-normn-nn-layernorm-d_model:end
        # @arch encoderlayer.self-normn-nn-layernorm-d_model.2:start
        self.norm2 = nn.LayerNorm(d_model)
        # @arch encoderlayer.self-normn-nn-layernorm-d_model.2:end

    # @arch encoderlayer.def-forward-self-x-src_mask-none:start
    def forward(self, x, src_mask=None):
    # @arch encoderlayer.def-forward-self-x-src_mask-none:end
        # Apply self-attention with residual normalization: (batch, steps, d_model).
        attn = self.self_attn(x, x, x, src_mask)  # (batch, steps, d_model)
        # @arch encoderlayer.forward.attn_residual-x-attn:start
        attn_residual = x + attn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        # @arch encoderlayer.forward.attn_residual-x-attn:end
        # @arch transformer.encoder.norm1:start
        x = self.norm1(attn_residual)  # (batch, steps, d_model)
        # @arch transformer.encoder.norm1:end

        # Apply feed-forward block with residual normalization.
        # @arch encoderlayer.forward.ffn-self-ffn-x:start
        ffn = self.ffn(x)  # (batch, steps, d_model)
        # @arch encoderlayer.forward.ffn-self-ffn-x:end
        # @arch transformer.encoder.ffn_residual:start
        ffn_residual = x + ffn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        # @arch transformer.encoder.ffn_residual:end
        # @arch transformer.encoder.norm2:start
        out = self.norm2(ffn_residual)  # (batch, steps, d_model)
        # @arch transformer.encoder.norm2:end
        # @arch encoderlayer.forward.return-out:start
        return out  # (batch, steps, d_model)
        # @arch encoderlayer.forward.return-out:end


# %% [notebook-only]
example_encoder_layer = EncoderLayer(d_model=8, nhead=2, d_ff=32)
encoder_input = torch.randn(2, 4, 8)  # -> (2, 4, 8)
example_encoder_output = example_encoder_layer(encoder_input)  # (2, 4, 8)
print(example_encoder_output.shape)


# %%
# @arch class-decoderlayer-nn-module:start
class DecoderLayer(nn.Module):
# @arch class-decoderlayer-nn-module:end
    # @arch decoderlayer.def-__init__:start
    def __init__(
    # @arch decoderlayer.def-__init__:end
        self,
        d_model=512,  # Embedding width.
        # @arch decoderlayer.nhead-n:start
        nhead=8,  # Number of attention heads.
        # @arch decoderlayer.nhead-n:end
        d_ff=2048  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register masked attention, cross-attention, feed-forward, and normalization layers.
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            # @arch decoderlayer.nn-linear-d_model-d_ff:start
            nn.Linear(d_model, d_ff),
            # @arch decoderlayer.nn-linear-d_model-d_ff:end
            # @arch decoderlayer.nn-relu:start
            nn.ReLU(),
            # @arch decoderlayer.nn-relu:end
            # @arch decoderlayer.nn-linear-d_ff-d_model:start
            nn.Linear(d_ff, d_model),
            # @arch decoderlayer.nn-linear-d_ff-d_model:end
        # @arch decoderlayer.code.4:start
        )
        # @arch decoderlayer.code.4:end
        # @arch decoderlayer.self-normn-nn-layernorm-d_model:start
        self.norm1 = nn.LayerNorm(d_model)
        # @arch decoderlayer.self-normn-nn-layernorm-d_model:end
        # @arch decoderlayer.self-normn-nn-layernorm-d_model.2:start
        self.norm2 = nn.LayerNorm(d_model)
        # @arch decoderlayer.self-normn-nn-layernorm-d_model.2:end
        # @arch decoderlayer.self-normn-nn-layernorm-d_model.3:start
        self.norm3 = nn.LayerNorm(d_model)
        # @arch decoderlayer.self-normn-nn-layernorm-d_model.3:end

    # @arch decoderlayer.def-forward-self-x-memory-tgt_mask-none:start
    def forward(self, x, memory, tgt_mask=None):
    # @arch decoderlayer.def-forward-self-x-memory-tgt_mask-none:end
        # Apply masked self-attention with residual normalization.
        # @arch decoderlayer.forward.masked-self-self_attn-x-x-x-tgt_mask:start
        masked = self.self_attn(x, x, x, tgt_mask)  # (batch, target_steps, d_model)
        # @arch decoderlayer.forward.masked-self-self_attn-x-x-x-tgt_mask:end
        masked_residual = x + masked  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.forward.x-self-normn-masked_residual:start
        x = self.norm1(masked_residual)  # (batch, target_steps, d_model)
        # @arch decoderlayer.forward.x-self-normn-masked_residual:end

        # Attend over encoder memory with residual normalization.
        # @arch decoderlayer.forward.cross-self-cross_attn-x-memory-memory:start
        cross = self.cross_attn(x, memory, memory)  # (batch, target_steps, d_model), (batch, source_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.forward.cross-self-cross_attn-x-memory-memory:end
        # @arch decoderlayer.forward.cross_residual-x-cross:start
        cross_residual = x + cross  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.forward.cross_residual-x-cross:end
        # @arch transformer.decoder.norm2:start
        x = self.norm2(cross_residual)  # (batch, target_steps, d_model)
        # @arch transformer.decoder.norm2:end

        # Apply feed-forward block with residual normalization.
        # @arch decoderlayer.forward.ffn-self-ffn-x:start
        ffn = self.ffn(x)  # (batch, target_steps, d_model)
        # @arch decoderlayer.forward.ffn-self-ffn-x:end
        # @arch decoderlayer.forward.ffn_residual-x-ffn:start
        ffn_residual = x + ffn  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        # @arch decoderlayer.forward.ffn_residual-x-ffn:end
        # @arch transformer.decoder.norm3:start
        out = self.norm3(ffn_residual)  # (batch, target_steps, d_model)
        # @arch transformer.decoder.norm3:end
        return out  # (batch, target_steps, d_model)


# %% [notebook-only]
decoder_layer = DecoderLayer(d_model=8, nhead=2, d_ff=32)
decoder_input = torch.randn(2, 4, 8)  # -> (2, 4, 8)
encoder_memory = torch.randn(2, 5, 8)  # -> (2, 5, 8)
mask_values = torch.ones(4, 4)  # -> (4, 4)
mask_values = mask_values * float('-inf')  # (4, 4)
target_mask = torch.triu(mask_values, diagonal=1)  # (4, 4)
example_decoder_output = decoder_layer(decoder_input, encoder_memory, target_mask)  # (2, 4, 8)
print(example_decoder_output.shape)


# %%
# @arch class-transformer-nn-module:start
class Transformer(nn.Module):
# @arch class-transformer-nn-module:end
    # @arch transformer.def-__init__:start
    def __init__(
    # @arch transformer.def-__init__:end
        # @arch transformer.self:start
        self,
        # @arch transformer.self:end
        vocab_size=37000,  # Source and target vocabulary size.
        d_model=512,  # Embedding width.
        nhead=8,  # Number of attention heads.
        num_layers=6  # Number of encoder and decoder layers.
    ):
        super().__init__()

        # Register token embeddings, shared position encoder, stacks, and generator.
        self.src_embed = nn.Embedding(vocab_size, d_model)
        # @arch transformer.self-tgt_embed-nn-embedding-vocab_size-d_model:start
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        # @arch transformer.self-tgt_embed-nn-embedding-vocab_size-d_model:end
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead) for _ in range(num_layers)])
        # @arch transformer.self-generator-nn-linear-d_model-vocab_size:start
        self.generator = nn.Linear(d_model, vocab_size)
        # @arch transformer.self-generator-nn-linear-d_model-vocab_size:end

    # @arch transformer.def-forward-self-src_ids-tgt_ids-tgt_mask:start
    def forward(self, src_ids, tgt_ids, tgt_mask):
    # @arch transformer.def-forward-self-src_ids-tgt_ids-tgt_mask:end
        # Embed and encode the source tokens: (batch, source_steps) -> memory.
        # @arch transformer.forward.src_embeddings-self-src_embed-src_ids:start
        src_embeddings = self.src_embed(src_ids)  # (batch, source_steps) -> (batch, source_steps, d_model)
        # @arch transformer.forward.src_embeddings-self-src_embed-src_ids:end
        # @arch transformer.forward.memory-self-pos-src_embeddings:start
        memory = self.pos(src_embeddings)  # (batch, source_steps, d_model)
        # @arch transformer.forward.memory-self-pos-src_embeddings:end
        for layer in self.encoder:
            memory = layer(memory)  # (batch, source_steps, d_model)

        # Embed target tokens and decode against source memory.
        # @arch transformer.forward.tgt_embeddings-self-tgt_embed-tgt_ids:start
        tgt_embeddings = self.tgt_embed(tgt_ids)  # (batch, target_steps) -> (batch, target_steps, d_model)
        # @arch transformer.forward.tgt_embeddings-self-tgt_embed-tgt_ids:end
        # @arch transformer.forward.x-self-pos-tgt_embeddings:start
        x = self.pos(tgt_embeddings)  # (batch, target_steps, d_model)
        # @arch transformer.forward.x-self-pos-tgt_embeddings:end
        # @arch transformer.forward.for-layer-in-self-decoder:start
        for layer in self.decoder:
        # @arch transformer.forward.for-layer-in-self-decoder:end
            x = layer(x, memory, tgt_mask)  # (batch, target_steps, d_model)

        # Project decoder states to vocabulary logits.
        # @arch transformer.forward.logits-self-generator-x:start
        logits = self.generator(x)  # (batch, target_steps, d_model) -> (batch, target_steps, vocab_size)
        # @arch transformer.forward.logits-self-generator-x:end
        # @arch transformer.forward.return-logits:start
        return logits  # (batch, target_steps, vocab_size)
        # @arch transformer.forward.return-logits:end


# %% [notebook-only]
# Create and run a sample translation batch.
example_model = Transformer(vocab_size=37000)
src_ids = torch.randint(0, 37000, (2, 16))  # -> (2, 16)
tgt_ids = torch.randint(0, 37000, (2, 16))  # -> (2, 16)

# Build a causal target mask: (16, 16).
mask_values = torch.ones(16, 16)  # -> (16, 16)
mask_values = mask_values * float('-inf')  # (16, 16)
tgt_mask = torch.triu(mask_values, diagonal=1)  # (16, 16)
example_logits = example_model(src_ids, tgt_ids, tgt_mask)  # (2, 16), (2, 16), (16, 16) -> (2, 16, 37000)
print(example_logits.shape)


# %%
# Train on a tiny copy-style token batch.
model = Transformer(vocab_size=20, d_model=16, nhead=4, num_layers=1)
src_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
tgt_ids = torch.tensor([[0, 1, 2, 3], [0, 4, 3, 2]])  # -> (2, 4)
train_targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
mask_values = torch.ones(4, 4)  # -> (4, 4)
mask_values = mask_values * float('-inf')  # (4, 4)
tgt_mask = torch.triu(mask_values, diagonal=1)  # (4, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
# @arch for-step-in-range-n:start
for step in range(3):
# @arch for-step-in-range-n:end
    # @arch optimizer-zero_grad:start
    optimizer.zero_grad()
    # @arch optimizer-zero_grad:end
    # @arch logits-model-src_ids-tgt_ids-tgt_mask:start
    logits = model(src_ids, tgt_ids, tgt_mask)  # (2, 4), (2, 4), (4, 4) -> (2, 4, 20)
    # @arch logits-model-src_ids-tgt_ids-tgt_mask:end
    vocab_size = logits.size(-1)  # (2, 4, 20) -> scalar
    flat_logits = logits.reshape(-1, vocab_size)  # (2, 4, 20) -> (8, 20)
    flat_targets = train_targets.reshape(-1)  # (2, 4) -> (8)
    loss = criterion(flat_logits, flat_targets)  # (8, 20), (8) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
