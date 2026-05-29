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


class BertLayer(nn.Module):
    def __init__(
        self,
        hidden_size=768,  # Embedding width.
        num_heads=12,  # Number of attention heads.
        intermediate_size=3072  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register attention, feed-forward, normalization, and dropout layers.
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
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
        attn, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
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

# mlm_logits: (2, 16, 30522), pooled: (2, 768)
