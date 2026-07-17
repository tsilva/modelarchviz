# %%
import torch
import torch.nn as nn


# %%
# @arch class-bertembeddings-nn-module:start
class BertEmbeddings(nn.Module):
# @arch class-bertembeddings-nn-module:end
    def __init__(
        self,
        vocab_size=30522,  # Number of token ids.
        hidden_size=768,  # Embedding width.
        max_position=512  # Maximum sequence length.
    ):
        super().__init__()

        # Register token, position, segment, normalization, and dropout layers.
        # @arch bertembeddings.self-word_embeddings-nn-embedding-vocab_size-hidden_size:start
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        # @arch bertembeddings.self-word_embeddings-nn-embedding-vocab_size-hidden_size:end
        # @arch bertembeddings.self-position_embeddings-nn-embedding-max_position-hidden_size:start
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        # @arch bertembeddings.self-position_embeddings-nn-embedding-max_position-hidden_size:end
        # @arch bertembeddings.self-token_type_embeddings-nn-embedding-n-hidden_size:start
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        # @arch bertembeddings.self-token_type_embeddings-nn-embedding-n-hidden_size:end
        # @arch bertembeddings.self-norm-nn-layernorm-hidden_size:start
        self.norm = nn.LayerNorm(hidden_size)
        # @arch bertembeddings.self-norm-nn-layernorm-hidden_size:end
        # @arch bertembeddings.self-dropout-nn-dropout-n:start
        self.dropout = nn.Dropout(0.1)
        # @arch bertembeddings.self-dropout-nn-dropout-n:end

    # @arch bertembeddings.def-forward-self-input_ids-token_type_ids:start
    def forward(self, input_ids, token_type_ids):
    # @arch bertembeddings.def-forward-self-input_ids-token_type_ids:end
        # Combine token, position, and segment embeddings: (batch, steps) -> (batch, steps, hidden_size).
        # @arch bertembeddings.forward.positions-torch-arange-input_ids-size-n-device-input_ids-device:start
        positions = torch.arange(input_ids.size(1), device=input_ids.device)  # -> (steps)
        # @arch bertembeddings.forward.positions-torch-arange-input_ids-size-n-device-input_ids-device:end
        # @arch bertembeddings.forward.x-self-word_embeddings-input_ids:start
        x = self.word_embeddings(input_ids)  # (batch, steps) -> (batch, steps, hidden_size)
        # @arch bertembeddings.forward.x-self-word_embeddings-input_ids:end
        # @arch bertembeddings.forward.position_embeddings-self-position_embeddings-positions:start
        position_embeddings = self.position_embeddings(positions)  # (steps) -> (steps, hidden_size)
        # @arch bertembeddings.forward.position_embeddings-self-position_embeddings-positions:end
        # @arch bertembeddings.forward.position_embeddings-position_embeddings-none:start
        position_embeddings = position_embeddings[None, :, :]  # (steps, hidden_size) -> (1, steps, hidden_size)
        # @arch bertembeddings.forward.position_embeddings-position_embeddings-none:end
        # @arch bertembeddings.forward.x-x-position_embeddings:start
        x = x + position_embeddings  # (batch, steps, hidden_size)
        # @arch bertembeddings.forward.x-x-position_embeddings:end
        # @arch bertembeddings.forward.x-x-self-token_type_embeddings-token_type_ids:start
        x = x + self.token_type_embeddings(token_type_ids)  # (batch, steps, hidden_size)
        # @arch bertembeddings.forward.x-x-self-token_type_embeddings-token_type_ids:end

        # Normalize and regularize embeddings while preserving shape.
        # @arch bertembeddings.forward.x-self-norm-x:start
        x = self.norm(x)  # (batch, steps, hidden_size)
        # @arch bertembeddings.forward.x-self-norm-x:end
        # @arch bertembeddings.forward.x-self-dropout-x:start
        x = self.dropout(x)  # (batch, steps, hidden_size)
        # @arch bertembeddings.forward.x-self-dropout-x:end
        # @arch bertembeddings.forward.return-x:start
        return x  # (batch, steps, hidden_size)
        # @arch bertembeddings.forward.return-x:end


# %% [notebook-only]
# Create and run the embedding block: (2, 4) -> (2, 4, 12).
embeddings = BertEmbeddings(vocab_size=20, hidden_size=12, max_position=8)
example_input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
example_token_type_ids = torch.zeros((2, 4), dtype=torch.long)  # -> (2, 4)
example_embedded = embeddings(example_input_ids, example_token_type_ids)  # (2, 4), (2, 4) -> (2, 4, 12)
print("embedded shape:", example_embedded.shape)

# %%
# @arch class-bertselfattention-nn-module:start
class BertSelfAttention(nn.Module):
# @arch class-bertselfattention-nn-module:end
    def __init__(
        self,
        hidden_size=768,  # Embedding width.
        num_heads=12  # Number of attention heads.
    ):
        super().__init__()

        # Register explicit Q/K/V projections and the output projection.
        # @arch bertselfattention.self-num_heads-num_heads:start
        self.num_heads = num_heads
        # @arch bertselfattention.self-num_heads-num_heads:end
        # @arch bertselfattention.self-head_dim-hidden_size-num_heads:start
        self.head_dim = hidden_size // num_heads
        # @arch bertselfattention.self-head_dim-hidden_size-num_heads:end
        # @arch bertselfattention.self-q_proj-nn-linear-hidden_size-hidden_size:start
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        # @arch bertselfattention.self-q_proj-nn-linear-hidden_size-hidden_size:end
        # @arch bertselfattention.self-k_proj-nn-linear-hidden_size-hidden_size:start
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        # @arch bertselfattention.self-k_proj-nn-linear-hidden_size-hidden_size:end
        # @arch bertselfattention.self-v_proj-nn-linear-hidden_size-hidden_size:start
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        # @arch bertselfattention.self-v_proj-nn-linear-hidden_size-hidden_size:end
        # @arch bertselfattention.self-out_proj-nn-linear-hidden_size-hidden_size:start
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # @arch bertselfattention.self-out_proj-nn-linear-hidden_size-hidden_size:end

    # @arch bertselfattention.def-forward-self-x-attention_mask-none:start
    def forward(self, x, attention_mask=None):
    # @arch bertselfattention.def-forward-self-x-attention_mask-none:end
        # Project token states into per-head query, key, and value tensors.
        # @arch bertselfattention.forward.batch_size-x-size-n:start
        batch_size = x.size(0)  # (batch, steps, hidden_size) -> scalar
        # @arch bertselfattention.forward.batch_size-x-size-n:end
        # @arch bertselfattention.forward.steps-x-size-n:start
        steps = x.size(1)  # (batch, steps, hidden_size) -> scalar
        # @arch bertselfattention.forward.steps-x-size-n:end
        # @arch bertselfattention.forward.q-self-q_proj-x:start
        q = self.q_proj(x)  # (batch, steps, hidden_size)
        # @arch bertselfattention.forward.q-self-q_proj-x:end
        # @arch bertselfattention.forward.k-self-k_proj-x:start
        k = self.k_proj(x)  # (batch, steps, hidden_size)
        # @arch bertselfattention.forward.k-self-k_proj-x:end
        # @arch bertselfattention.forward.v-self-v_proj-x:start
        v = self.v_proj(x)  # (batch, steps, hidden_size)
        # @arch bertselfattention.forward.v-self-v_proj-x:end

        # Split model width across heads: (batch, steps, hidden_size) -> (batch, heads, steps, head_dim).
        # @arch bertselfattention.forward.q-q-view-batch_size-steps-self-num_heads-self-head_dim:start
        q = q.view(batch_size, steps, self.num_heads, self.head_dim)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.forward.q-q-view-batch_size-steps-self-num_heads-self-head_dim:end
        # @arch bertselfattention.forward.q-q-transpose-n-n:start
        q = q.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.forward.q-q-transpose-n-n:end
        # @arch bertselfattention.forward.k-k-view-batch_size-steps-self-num_heads-self-head_dim:start
        k = k.view(batch_size, steps, self.num_heads, self.head_dim)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.forward.k-k-view-batch_size-steps-self-num_heads-self-head_dim:end
        # @arch bertselfattention.forward.k-k-transpose-n-n:start
        k = k.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.forward.k-k-transpose-n-n:end
        # @arch bertselfattention.forward.v-v-view-batch_size-steps-self-num_heads-self-head_dim:start
        v = v.view(batch_size, steps, self.num_heads, self.head_dim)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.forward.v-v-view-batch_size-steps-self-num_heads-self-head_dim:end
        # @arch bertselfattention.forward.v-v-transpose-n-n:start
        v = v.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.forward.v-v-transpose-n-n:end

        # Compute scaled dot-product attention and mask padded keys.
        # @arch bertselfattention.forward.key_transpose-k-transpose-n-n:start
        key_transpose = k.transpose(-2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        # @arch bertselfattention.forward.key_transpose-k-transpose-n-n:end
        # @arch bertselfattention.forward.scores-q-key_transpose:start
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        # @arch bertselfattention.forward.scores-q-key_transpose:end
        # @arch bertselfattention.forward.scale-self-head_dim-n:start
        scale = self.head_dim ** -0.5  # scalar
        # @arch bertselfattention.forward.scale-self-head_dim-n:end
        # @arch bertselfattention.forward.attn_scores-scores-scale:start
        attn_scores = scores * scale  # (batch, heads, steps, steps)
        # @arch bertselfattention.forward.attn_scores-scores-scale:end
        # @arch bertselfattention.forward.if-attention_mask-is-not-none:start
        if attention_mask is not None:
        # @arch bertselfattention.forward.if-attention_mask-is-not-none:end
            # @arch bertselfattention.forward.mask-attention_mask-none-none:start
            mask = attention_mask[:, None, None, :]  # (batch, steps) -> (batch, 1, 1, steps)
            # @arch bertselfattention.forward.mask-attention_mask-none-none:end
            # @arch bertselfattention.forward.attn_scores-attn_scores-masked_fill-mask-nen:start
            attn_scores = attn_scores.masked_fill(mask, -1e9)  # (batch, heads, steps, steps)
            # @arch bertselfattention.forward.attn_scores-attn_scores-masked_fill-mask-nen:end
        # @arch bertselfattention.forward.attn_weights-torch-softmax-attn_scores-dim-n:start
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, heads, steps, steps)
        # @arch bertselfattention.forward.attn_weights-torch-softmax-attn_scores-dim-n:end

        # Mix values, merge heads, and project back to hidden width.
        # @arch bertselfattention.forward.context-attn_weights-v:start
        context = attn_weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        # @arch bertselfattention.forward.context-attn_weights-v:end
        # @arch bertselfattention.forward.context-context-transpose-n-n:start
        context = context.transpose(1, 2)  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        # @arch bertselfattention.forward.context-context-transpose-n-n:end
        # @arch bertselfattention.forward.context-context-contiguous:start
        context = context.contiguous()  # (batch, steps, heads, head_dim)
        # @arch bertselfattention.forward.context-context-contiguous:end
        # @arch bertselfattention.forward.merged-context-view-batch_size-steps-self-num_heads-self-head_dim:start
        merged = context.view(batch_size, steps, self.num_heads * self.head_dim)  # (batch, steps, heads, head_dim) -> (batch, steps, hidden_size)
        # @arch bertselfattention.forward.merged-context-view-batch_size-steps-self-num_heads-self-head_dim:end
        # @arch bertselfattention.forward.out-self-out_proj-merged:start
        out = self.out_proj(merged)  # (batch, steps, hidden_size)
        # @arch bertselfattention.forward.out-self-out_proj-merged:end
        # @arch bertselfattention.forward.return-out:start
        return out  # (batch, steps, hidden_size)
        # @arch bertselfattention.forward.return-out:end


# %% [notebook-only]
# Create and run one BERT self-attention block: (2, 4, 12) -> (2, 4, 12).
example_attention = BertSelfAttention(hidden_size=12, num_heads=3)
example_hidden_states = torch.randn(2, 4, 12)  # -> (2, 4, 12)
attention_mask = torch.zeros((2, 4), dtype=torch.bool)  # -> (2, 4)
example_attended = example_attention(example_hidden_states, attention_mask)  # (2, 4, 12), (2, 4) -> (2, 4, 12)
print("attended shape:", example_attended.shape)

# %%
# @arch class-bertlayer-nn-module:start
class BertLayer(nn.Module):
# @arch class-bertlayer-nn-module:end
    # @arch bertlayer.def-__init__:start
    def __init__(
    # @arch bertlayer.def-__init__:end
        # @arch bertlayer.self:start
        self,
        # @arch bertlayer.self:end
        # @arch bertlayer.hidden_size-n:start
        hidden_size=768,  # Embedding width.
        # @arch bertlayer.hidden_size-n:end
        # @arch bertlayer.num_heads-n:start
        num_heads=12,  # Number of attention heads.
        # @arch bertlayer.num_heads-n:end
        # @arch bertlayer.intermediate_size-n:start
        intermediate_size=3072  # Feed-forward hidden width.
        # @arch bertlayer.intermediate_size-n:end
    # @arch bertlayer.code:start
    ):
    # @arch bertlayer.code:end
        # @arch bertlayer.super-__init__:start
        super().__init__()
        # @arch bertlayer.super-__init__:end

        # Register attention, feed-forward, normalization, and dropout layers.
        # @arch bertlayer.self-self_attn-bertselfattention-hidden_size-num_heads:start
        self.self_attn = BertSelfAttention(hidden_size, num_heads)
        # @arch bertlayer.self-self_attn-bertselfattention-hidden_size-num_heads:end
        # @arch bertlayer.self-attn_norm-nn-layernorm-hidden_size:start
        self.attn_norm = nn.LayerNorm(hidden_size)
        # @arch bertlayer.self-attn_norm-nn-layernorm-hidden_size:end
        # @arch bertlayer.self-ffn-nn-sequential:start
        self.ffn = nn.Sequential(
        # @arch bertlayer.self-ffn-nn-sequential:end
            # @arch bertlayer.nn-linear-hidden_size-intermediate_size:start
            nn.Linear(hidden_size, intermediate_size),
            # @arch bertlayer.nn-linear-hidden_size-intermediate_size:end
            # @arch bertlayer.nn-gelu:start
            nn.GELU(),
            # @arch bertlayer.nn-gelu:end
            # @arch bertlayer.nn-linear-intermediate_size-hidden_size:start
            nn.Linear(intermediate_size, hidden_size),
            # @arch bertlayer.nn-linear-intermediate_size-hidden_size:end
        # @arch bertlayer.code.4:start
        )
        # @arch bertlayer.code.4:end
        # @arch bertlayer.self-ffn_norm-nn-layernorm-hidden_size:start
        self.ffn_norm = nn.LayerNorm(hidden_size)
        # @arch bertlayer.self-ffn_norm-nn-layernorm-hidden_size:end
        # @arch bertlayer.self-dropout-nn-dropout-n:start
        self.dropout = nn.Dropout(0.1)
        # @arch bertlayer.self-dropout-nn-dropout-n:end

    # @arch bertlayer.def-forward-self-x-attention_mask-none:start
    def forward(self, x, attention_mask=None):
    # @arch bertlayer.def-forward-self-x-attention_mask-none:end
        # Apply self-attention with residual normalization: (batch, steps, hidden_size).
        # @arch bertlayer.forward.attn-self-self_attn-x-attention_mask:start
        attn = self.self_attn(x, attention_mask)  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.attn-self-self_attn-x-attention_mask:end
        # @arch bertlayer.forward.attn-self-dropout-attn:start
        attn = self.dropout(attn)  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.attn-self-dropout-attn:end
        # @arch bertlayer.forward.attn_residual-x-attn:start
        attn_residual = x + attn  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.attn_residual-x-attn:end
        # @arch bertlayer.forward.x-self-attn_norm-attn_residual:start
        x = self.attn_norm(attn_residual)  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.x-self-attn_norm-attn_residual:end

        # Apply feed-forward block with residual normalization.
        # @arch bertlayer.forward.ffn-self-ffn-x:start
        ffn = self.ffn(x)  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.ffn-self-ffn-x:end
        # @arch bertlayer.forward.ffn-self-dropout-ffn:start
        ffn = self.dropout(ffn)  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.ffn-self-dropout-ffn:end
        # @arch bertlayer.forward.ffn_residual-x-ffn:start
        ffn_residual = x + ffn  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.ffn_residual-x-ffn:end
        # @arch bertlayer.forward.out-self-ffn_norm-ffn_residual:start
        out = self.ffn_norm(ffn_residual)  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.out-self-ffn_norm-ffn_residual:end
        # @arch bertlayer.forward.return-out:start
        return out  # (batch, steps, hidden_size)
        # @arch bertlayer.forward.return-out:end


# %% [notebook-only]
# Create and run one encoder layer: (2, 4, 12) -> (2, 4, 12).
layer = BertLayer(hidden_size=12, num_heads=3, intermediate_size=24)
example_hidden_states = torch.randn(2, 4, 12)  # -> (2, 4, 12)
attention_mask = torch.zeros((2, 4), dtype=torch.bool)  # -> (2, 4)
example_layer_output = layer(example_hidden_states, attention_mask)  # (2, 4, 12), (2, 4) -> (2, 4, 12)
print("layer_output shape:", example_layer_output.shape)

# %%
class BERTBase(nn.Module):
    def __init__(
        self,
        vocab_size=30522,  # Number of token ids.
        hidden_size=768,  # Embedding width.
        num_layers=12  # Number of encoder layers.
    ):
        super().__init__()

        # Register embedding, encoder stack, pooler, and masked-language-model head.
        # @arch bertbase.self-embeddings-bertembeddings-vocab_size-hidden_size:start
        self.embeddings = BertEmbeddings(vocab_size, hidden_size)
        # @arch bertbase.self-embeddings-bertembeddings-vocab_size-hidden_size:end
        # @arch bertbase.self-layers-nn-modulelist-bertlayer-hidden_size-for-_-in-range-num_layer:start
        self.layers = nn.ModuleList([BertLayer(hidden_size) for _ in range(num_layers)])
        # @arch bertbase.self-layers-nn-modulelist-bertlayer-hidden_size-for-_-in-range-num_layer:end
        # @arch bertbase.self-pooler-nn-linear-hidden_size-hidden_size:start
        self.pooler = nn.Linear(hidden_size, hidden_size)
        # @arch bertbase.self-pooler-nn-linear-hidden_size-hidden_size:end
        # @arch bertbase.self-mlm-nn-linear-hidden_size-vocab_size:start
        self.mlm = nn.Linear(hidden_size, vocab_size)
        # @arch bertbase.self-mlm-nn-linear-hidden_size-vocab_size:end

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        # Embed tokens and run the encoder stack.
        # @arch bertbase.forward.x-self-embeddings-input_ids-token_type_ids:start
        x = self.embeddings(input_ids, token_type_ids)  # (batch, steps) -> (batch, steps, hidden_size)
        # @arch bertbase.forward.x-self-embeddings-input_ids-token_type_ids:end
        # @arch bertbase.forward.for-layer-in-self-layers:start
        for layer in self.layers:
        # @arch bertbase.forward.for-layer-in-self-layers:end
            # @arch bertbase.forward.x-layer-x-attention_mask:start
            x = layer(x, attention_mask)  # (batch, steps, hidden_size)
            # @arch bertbase.forward.x-layer-x-attention_mask:end

        # Pool the CLS token and project sequence states to token logits.
        # @arch bertbase.forward.cls_token-x-n:start
        cls_token = x[:, 0]  # (batch, steps, hidden_size) -> (batch, hidden_size)
        # @arch bertbase.forward.cls_token-x-n:end
        # @arch bertbase.forward.pooled_projection-self-pooler-cls_token:start
        pooled_projection = self.pooler(cls_token)  # (batch, hidden_size)
        # @arch bertbase.forward.pooled_projection-self-pooler-cls_token:end
        # @arch bertbase.forward.pooled-torch-tanh-pooled_projection:start
        pooled = torch.tanh(pooled_projection)  # (batch, hidden_size)
        # @arch bertbase.forward.pooled-torch-tanh-pooled_projection:end
        # @arch bertbase.forward.mlm_logits-self-mlm-x:start
        mlm_logits = self.mlm(x)  # (batch, steps, hidden_size) -> (batch, steps, vocab_size)
        # @arch bertbase.forward.mlm_logits-self-mlm-x:end
        outputs = (mlm_logits, pooled)
        return outputs


# %% [notebook-only]
# Create and run a sample token batch.
example_model = BERTBase(vocab_size=30522)
example_input_ids = torch.randint(0, 30522, (2, 16))  # -> (2, 16)
example_token_type_ids = torch.zeros((2, 16), dtype=torch.long)  # -> (2, 16)
attention_mask = torch.zeros((2, 16), dtype=torch.bool)  # -> (2, 16)
example_outputs = example_model(example_input_ids, example_token_type_ids, attention_mask)  # (2, 16), (2, 16), (2, 16) -> tuple
mlm_logits = example_outputs[0]  # tuple -> (2, 16, 30522)
pooled = example_outputs[1]  # tuple -> (2, 768)
print("mlm logits shape:", mlm_logits.shape, "pooled shape:", pooled.shape)

# %%
# Train on a tiny masked-token prediction batch.
model = BERTBase(vocab_size=20, hidden_size=12, num_layers=1)
# @arch input_ids-torch-tensor-n-n-n-n-n-n-n-n:start
input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
# @arch input_ids-torch-tensor-n-n-n-n-n-n-n-n:end
# @arch token_type_ids-torch-zeros-n-n-dtype-torch-long:start
token_type_ids = torch.zeros((2, 4), dtype=torch.long)  # -> (2, 4)
# @arch token_type_ids-torch-zeros-n-n-dtype-torch-long:end
attention_mask = torch.zeros((2, 4), dtype=torch.bool)  # -> (2, 4)
train_targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])  # -> (2, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    # @arch outputs-model-input_ids-token_type_ids-attention_mask:start
    outputs = model(input_ids, token_type_ids, attention_mask)  # (2, 4), (2, 4), (2, 4) -> tuple
    # @arch outputs-model-input_ids-token_type_ids-attention_mask:end
    mlm_logits = outputs[0]  # tuple -> (2, 4, 20)
    flat_logits = mlm_logits.reshape(-1, mlm_logits.size(-1))  # (2, 4, 20) -> (8, 20)
    flat_targets = train_targets.reshape(-1)  # (2, 4) -> (8)
    loss = criterion(flat_logits, flat_targets)  # (8, 20), (8) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
