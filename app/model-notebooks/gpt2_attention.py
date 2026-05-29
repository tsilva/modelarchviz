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
import torch.nn.functional as F


class GPT2Small(nn.Module):
    def __init__(
        self,
        vocab_size,  # Number of token ids.
        n_ctx=1024,  # Maximum context length.
        n_embd=768  # Embedding width.
    ):
        super().__init__()

        # Register embeddings, transformer blocks, final norm, and language-model head.
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_ctx, n_embd)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, mask):
        # Combine token and position embeddings: (batch, steps) -> (batch, steps, n_embd).
        batch_size, step_count = input_ids.shape
        positions = torch.arange(step_count, device=input_ids.device)
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(positions)
        position_embeddings = position_embeddings[None, :, :]
        x = token_embeddings + position_embeddings
        x = self.drop(x)

        # Run the transformer block stack while preserving sequence shape.
        for block in self.blocks:
            x = block(x, mask)

        # Normalize final states and project to vocabulary logits.
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd=768,  # Embedding width.
        n_head=12  # Number of attention heads.
    ):
        super().__init__()

        # Register packed QKV projection and output projection.
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x, mask):
        # Project hidden states into query, key, and value tensors.
        batch_size, step_count, channel_count = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(channel_count, dim=2)
        head_dim = channel_count // self.n_head

        # Split heads: (batch, steps, channels) -> (batch, heads, steps, head_dim).
        q = q.view(batch_size, step_count, self.n_head, head_dim)
        q = q.transpose(1, 2)
        k = k.view(batch_size, step_count, self.n_head, head_dim)
        k = k.transpose(1, 2)
        v = v.view(batch_size, step_count, self.n_head, head_dim)
        v = v.transpose(1, 2)

        # Compute masked causal attention weights.
        key_transpose = k.transpose(-2, -1)
        scores = q @ key_transpose
        scale = k.size(-1) ** -0.5
        att = scores * scale
        mask_window = mask[:, :, :step_count, :step_count]
        att = att.masked_fill(mask_window == 0, float('-inf'))
        weights = F.softmax(att, dim=-1)

        # Merge heads back to the model width and project.
        y = weights @ v
        y = y.transpose(1, 2)
        y = y.contiguous()
        y = y.view(batch_size, step_count, channel_count)
        out = self.c_proj(y)
        return out


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        # Register pre-normalized attention and MLP sublayers.
        self.ln_1 = nn.LayerNorm(768)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(768)
        self.mlp = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )

    def forward(self, x, mask):
        # Apply causal attention with a residual connection.
        attn_input = self.ln_1(x)
        attn = self.attn(attn_input, mask)
        x = x + attn

        # Apply MLP with a residual connection.
        mlp_input = self.ln_2(x)
        mlp_out = self.mlp(mlp_input)
        x = x + mlp_out
        return x


# Create and run a sample token batch.
model = GPT2Small(vocab_size=50257)
test_input = torch.randint(0, 50257, (2, 16))

# Build a causal attention mask: (1, 1, 16, 16).
mask_values = torch.ones(16, 16)
mask = torch.tril(mask_values)
mask = mask.view(1, 1, 16, 16)
logits = model(test_input, mask)

# logits: (2, 16, 50257)
