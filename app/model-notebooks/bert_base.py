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
import torch
import torch.nn as nn


class BertEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size=30522,  # Number of token ids.
        hidden_size=768,  # Embedding width.
        max_position=512  # Maximum sequence length.
    ):
        super().__init__()

        # Register token, position, segment, normalization, and dropout layers.
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids):
        # Combine token, position, and segment embeddings: (batch, steps) -> (batch, steps, hidden_size).
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(positions)
        position_embeddings = position_embeddings[None, :, :]
        x = x + position_embeddings
        x = x + self.token_type_embeddings(token_type_ids)

        # Normalize and regularize embeddings while preserving shape.
        x = self.norm(x)
        x = self.dropout(x)
        return x


class BertSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size=768,  # Embedding width.
        num_heads=12  # Number of attention heads.
    ):
        super().__init__()

        # Register explicit Q/K/V projections and the output projection.
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, attention_mask=None):
        # Project token states into per-head query, key, and value tensors.
        batch_size = x.size(0)  # (batch, steps, hidden_size) -> scalar
        steps = x.size(1)  # (batch, steps, hidden_size) -> scalar
        q = self.q_proj(x)  # (batch, steps, hidden_size)
        k = self.k_proj(x)  # (batch, steps, hidden_size)
        v = self.v_proj(x)  # (batch, steps, hidden_size)

        # Split model width across heads: (batch, steps, hidden_size) -> (batch, heads, steps, head_dim).
        q = q.view(batch_size, steps, self.num_heads, self.head_dim)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        q = q.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        k = k.view(batch_size, steps, self.num_heads, self.head_dim)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        k = k.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        v = v.view(batch_size, steps, self.num_heads, self.head_dim)  # (batch, steps, hidden_size) -> (batch, steps, heads, head_dim)
        v = v.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)

        # Compute scaled dot-product attention and mask padded keys.
        key_transpose = k.transpose(-2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        scale = self.head_dim ** -0.5  # scalar
        attn_scores = scores * scale  # (batch, heads, steps, steps)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]  # (batch, steps) -> (batch, 1, 1, steps)
            attn_scores = attn_scores.masked_fill(mask, -1e9)  # (batch, heads, steps, steps)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, heads, steps, steps)

        # Mix values, merge heads, and project back to hidden width.
        context = attn_weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        context = context.transpose(1, 2)  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        context = context.contiguous()  # (batch, steps, heads, head_dim)
        merged = context.view(batch_size, steps, self.num_heads * self.head_dim)  # (batch, steps, heads, head_dim) -> (batch, steps, hidden_size)
        out = self.out_proj(merged)  # (batch, steps, hidden_size)
        return out  # (batch, steps, hidden_size)


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size=768,  # Embedding width.
        num_heads=12,  # Number of attention heads.
        intermediate_size=3072  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register attention, feed-forward, normalization, and dropout layers.
        self.self_attn = BertSelfAttention(hidden_size, num_heads)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask=None):
        # Apply self-attention with residual normalization: (batch, steps, hidden_size).
        attn = self.self_attn(x, attention_mask)
        attn = self.dropout(attn)
        attn_residual = x + attn
        x = self.attn_norm(attn_residual)

        # Apply feed-forward block with residual normalization.
        ffn = self.ffn(x)
        ffn = self.dropout(ffn)
        ffn_residual = x + ffn
        out = self.ffn_norm(ffn_residual)
        return out


class BERTBase(nn.Module):
    def __init__(
        self,
        vocab_size=30522,  # Number of token ids.
        hidden_size=768,  # Embedding width.
        num_layers=12  # Number of encoder layers.
    ):
        super().__init__()

        # Register embedding, encoder stack, pooler, and masked-language-model head.
        self.embeddings = BertEmbeddings(vocab_size, hidden_size)
        self.layers = nn.ModuleList([BertLayer(hidden_size) for _ in range(num_layers)])
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.mlm = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        # Embed tokens and run the encoder stack.
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Pool the CLS token and project sequence states to token logits.
        cls_token = x[:, 0]
        pooled_projection = self.pooler(cls_token)
        pooled = torch.tanh(pooled_projection)
        mlm_logits = self.mlm(x)
        outputs = (mlm_logits, pooled)
        return outputs


# Create and run a sample token batch.
model = BERTBase(vocab_size=30522)
input_ids = torch.randint(0, 30522, (2, 16))
token_type_ids = torch.zeros((2, 16), dtype=torch.long)
attention_mask = torch.zeros((2, 16), dtype=torch.bool)
outputs = model(input_ids, token_type_ids, attention_mask)
mlm_logits = outputs[0]
pooled = outputs[1]


# Train on a tiny masked-token prediction batch.
model = BERTBase(vocab_size=20, hidden_size=12, num_layers=1)
input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
token_type_ids = torch.zeros((2, 4), dtype=torch.long)
attention_mask = torch.zeros((2, 4), dtype=torch.bool)
train_targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(input_ids, token_type_ids, attention_mask)
    mlm_logits = outputs[0]
    flat_logits = mlm_logits.reshape(-1, mlm_logits.size(-1))
    flat_targets = train_targets.reshape(-1)
    loss = criterion(flat_logits, flat_targets)
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
