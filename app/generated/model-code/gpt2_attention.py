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
        batch_size, step_count = input_ids.shape  # (batch, steps) -> scalar, scalar
        positions = torch.arange(step_count, device=input_ids.device)  # -> (steps)
        token_embeddings = self.wte(input_ids)  # (batch, steps) -> (batch, steps, n_embd)
        position_embeddings = self.wpe(positions)  # (steps) -> (steps, n_embd)
        position_embeddings = position_embeddings[None, :, :]  # (steps, n_embd) -> (1, steps, n_embd)
        x = token_embeddings + position_embeddings  # (batch, steps, n_embd)
        x = self.drop(x)  # (batch, steps, n_embd)

        # Run the transformer block stack while preserving sequence shape.
        for block in self.blocks:
            x = block(x, mask)  # (batch, steps, n_embd)

        # Normalize final states and project to vocabulary logits.
        x = self.ln_f(x)  # (batch, steps, n_embd)
        logits = self.lm_head(x)  # (batch, steps, n_embd) -> (batch, steps, vocab_size)
        return logits  # (batch, steps, vocab_size)


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
        batch_size, step_count, channel_count = x.shape  # (batch, steps, channels) -> scalar, scalar, scalar
        qkv = self.c_attn(x)  # (batch, steps, channels) -> (batch, steps, 3 * channels)
        q, k, v = qkv.split(channel_count, dim=2)  # (batch, steps, 3 * channels) -> three (batch, steps, channels)
        head_dim = channel_count // self.n_head  # scalar

        # Split heads: (batch, steps, channels) -> (batch, heads, steps, head_dim).
        q = q.view(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        q = q.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        k = k.view(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        k = k.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        v = v.view(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        v = v.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)

        # Compute masked causal attention weights.
        key_transpose = k.transpose(-2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        scale = k.size(-1) ** -0.5  # (batch, heads, steps, head_dim) -> scalar
        att = scores * scale  # (batch, heads, steps, steps)
        mask_window = mask[:, :, :step_count, :step_count]  # (1, 1, max_steps, max_steps) -> (1, 1, steps, steps)
        att = att.masked_fill(mask_window == 0, float('-inf'))  # (batch, heads, steps, steps)
        weights = F.softmax(att, dim=-1)  # (batch, heads, steps, steps)

        # Merge heads back to the model width and project.
        y = weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        y = y.transpose(1, 2)  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        y = y.contiguous()  # (batch, steps, heads, head_dim)
        y = y.view(batch_size, step_count, channel_count)  # (batch, steps, heads, head_dim) -> (batch, steps, channels)
        out = self.c_proj(y)  # (batch, steps, channels)
        return out  # (batch, steps, channels)


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
        attn_input = self.ln_1(x)  # (batch, steps, 768)
        attn = self.attn(attn_input, mask)  # (batch, steps, 768)
        x = x + attn  # (batch, steps, 768)

        # Apply MLP with a residual connection.
        mlp_input = self.ln_2(x)  # (batch, steps, 768)
        mlp_out = self.mlp(mlp_input)  # (batch, steps, 768)
        x = x + mlp_out  # (batch, steps, 768)
        return x  # (batch, steps, 768)


# Create and run a sample token batch.
model = GPT2Small(vocab_size=50257)
test_input = torch.randint(0, 50257, (2, 16))  # -> (2, 16)

# Build a causal attention mask: (1, 1, 16, 16).
mask_values = torch.ones(16, 16)  # -> (16, 16)
mask = torch.tril(mask_values)  # (16, 16)
mask = mask.view(1, 1, 16, 16)  # (16, 16) -> (1, 1, 16, 16)
logits = model(test_input, mask)  # (2, 16), (1, 1, 16, 16) -> (2, 16, 50257)


# Train on a tiny next-token prediction batch.
model = GPT2Small(vocab_size=20)
input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
train_targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])  # -> (2, 4)
mask_values = torch.ones(4, 4)  # -> (4, 4)
mask = torch.tril(mask_values)  # (4, 4)
mask = mask.view(1, 1, 4, 4)  # (4, 4) -> (1, 1, 4, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(input_ids, mask)  # (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
    flat_logits = logits.reshape(-1, logits.size(-1))  # (2, 4, 20) -> (8, 20)
    flat_targets = train_targets.reshape(-1)  # (2, 4) -> (8)
    loss = criterion(flat_logits, flat_targets)  # (8, 20), (8) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
