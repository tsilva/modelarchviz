import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model=512,  # Embedding width.
        max_len=5000  # Maximum supported sequence length.
    ):
        super().__init__()

        # Build sinusoidal position table: (max_len, d_model).
        positions = torch.arange(max_len)  # -> (max_len)
        position = positions.unsqueeze(1)  # (max_len) -> (max_len, 1)
        even_indices = torch.arange(0, d_model, 2)  # -> (d_model // 2)
        log_base = torch.log(torch.tensor(10000.0))  # -> scalar
        scale = -log_base / d_model  # scalar
        div_term = torch.exp(even_indices * scale)  # (d_model // 2)
        pe = torch.zeros(max_len, d_model)  # -> (max_len, d_model)
        sin_values = torch.sin(position * div_term)  # (max_len, 1), (d_model // 2) -> (max_len, d_model // 2)
        cos_values = torch.cos(position * div_term)  # (max_len, 1), (d_model // 2) -> (max_len, d_model // 2)
        pe[:, 0::2] = sin_values  # (max_len, d_model), (max_len, d_model // 2) -> (max_len, d_model)
        pe[:, 1::2] = cos_values  # (max_len, d_model), (max_len, d_model // 2) -> (max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encodings to embeddings: (batch, steps, d_model).
        seq_len = x.size(1)  # (batch, steps, d_model) -> scalar
        position_encoding = self.pe[:seq_len]  # (max_len, d_model) -> (steps, d_model)
        encoded = x + position_encoding  # (batch, steps, d_model), (steps, d_model) -> (batch, steps, d_model)
        return encoded  # (batch, steps, d_model)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model=512,  # Embedding width.
        nhead=8  # Number of attention heads.
    ):
        super().__init__()

        # Register explicit Q/K/V projections and the output projection.
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        # Project inputs into per-head query, key, and value tensors.
        batch_size = query.size(0)  # (batch, query_steps, d_model) -> scalar
        query_steps = query.size(1)  # (batch, query_steps, d_model) -> scalar
        key_steps = key.size(1)  # (batch, key_steps, d_model) -> scalar
        q = self.q_proj(query)  # (batch, query_steps, d_model)
        k = self.k_proj(key)  # (batch, key_steps, d_model)
        v = self.v_proj(value)  # (batch, key_steps, d_model)

        # Split model width across heads: (batch, steps, d_model) -> (batch, heads, steps, head_dim).
        q = q.view(batch_size, query_steps, self.nhead, self.head_dim)  # (batch, query_steps, d_model) -> (batch, query_steps, heads, head_dim)
        q = q.transpose(1, 2)  # (batch, query_steps, heads, head_dim) -> (batch, heads, query_steps, head_dim)
        k = k.view(batch_size, key_steps, self.nhead, self.head_dim)  # (batch, key_steps, d_model) -> (batch, key_steps, heads, head_dim)
        k = k.transpose(1, 2)  # (batch, key_steps, heads, head_dim) -> (batch, heads, key_steps, head_dim)
        v = v.view(batch_size, key_steps, self.nhead, self.head_dim)  # (batch, key_steps, d_model) -> (batch, key_steps, heads, head_dim)
        v = v.transpose(1, 2)  # (batch, key_steps, heads, head_dim) -> (batch, heads, key_steps, head_dim)

        # Compute scaled dot-product attention scores.
        key_transpose = k.transpose(-2, -1)  # (batch, heads, key_steps, head_dim) -> (batch, heads, head_dim, key_steps)
        scores = q @ key_transpose  # (batch, heads, query_steps, head_dim), (batch, heads, head_dim, key_steps) -> (batch, heads, query_steps, key_steps)
        scale = self.head_dim ** -0.5  # scalar
        attn_scores = scores * scale  # (batch, heads, query_steps, key_steps)
        if attn_mask is not None:
            mask = attn_mask[None, None, :, :]  # (query_steps, key_steps) -> (1, 1, query_steps, key_steps)
            attn_scores = attn_scores + mask  # (batch, heads, query_steps, key_steps)
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, heads, query_steps, key_steps)

        # Mix values, merge heads, and project back to model width.
        context = attn_weights @ v  # (batch, heads, query_steps, key_steps), (batch, heads, key_steps, head_dim) -> (batch, heads, query_steps, head_dim)
        context = context.transpose(1, 2)  # (batch, heads, query_steps, head_dim) -> (batch, query_steps, heads, head_dim)
        context = context.contiguous()  # (batch, query_steps, heads, head_dim)
        merged = context.view(batch_size, query_steps, self.nhead * self.head_dim)  # (batch, query_steps, heads, head_dim) -> (batch, query_steps, d_model)
        out = self.out_proj(merged)  # (batch, query_steps, d_model)
        return out  # (batch, query_steps, d_model)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,  # Embedding width.
        nhead=8,  # Number of attention heads.
        d_ff=2048  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register self-attention, feed-forward, and normalization layers.
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # Apply self-attention with residual normalization: (batch, steps, d_model).
        attn = self.self_attn(x, x, x, src_mask)  # (batch, steps, d_model)
        attn_residual = x + attn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        x = self.norm1(attn_residual)  # (batch, steps, d_model)

        # Apply feed-forward block with residual normalization.
        ffn = self.ffn(x)  # (batch, steps, d_model)
        ffn_residual = x + ffn  # (batch, steps, d_model), (batch, steps, d_model) -> (batch, steps, d_model)
        out = self.norm2(ffn_residual)  # (batch, steps, d_model)
        return out  # (batch, steps, d_model)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,  # Embedding width.
        nhead=8,  # Number of attention heads.
        d_ff=2048  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register masked attention, cross-attention, feed-forward, and normalization layers.
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask=None):
        # Apply masked self-attention with residual normalization.
        masked = self.self_attn(x, x, x, tgt_mask)  # (batch, target_steps, d_model)
        masked_residual = x + masked  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        x = self.norm1(masked_residual)  # (batch, target_steps, d_model)

        # Attend over encoder memory with residual normalization.
        cross = self.cross_attn(x, memory, memory)  # (batch, target_steps, d_model), (batch, source_steps, d_model) -> (batch, target_steps, d_model)
        cross_residual = x + cross  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        x = self.norm2(cross_residual)  # (batch, target_steps, d_model)

        # Apply feed-forward block with residual normalization.
        ffn = self.ffn(x)  # (batch, target_steps, d_model)
        ffn_residual = x + ffn  # (batch, target_steps, d_model), (batch, target_steps, d_model) -> (batch, target_steps, d_model)
        out = self.norm3(ffn_residual)  # (batch, target_steps, d_model)
        return out  # (batch, target_steps, d_model)


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size=37000,  # Source and target vocabulary size.
        d_model=512,  # Embedding width.
        nhead=8,  # Number of attention heads.
        num_layers=6  # Number of encoder and decoder layers.
    ):
        super().__init__()

        # Register token embeddings, shared position encoder, stacks, and generator.
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead) for _ in range(num_layers)])
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_ids, tgt_mask):
        # Embed and encode the source tokens: (batch, source_steps) -> memory.
        src_embeddings = self.src_embed(src_ids)  # (batch, source_steps) -> (batch, source_steps, d_model)
        memory = self.pos(src_embeddings)  # (batch, source_steps, d_model)
        for layer in self.encoder:
            memory = layer(memory)  # (batch, source_steps, d_model)

        # Embed target tokens and decode against source memory.
        tgt_embeddings = self.tgt_embed(tgt_ids)  # (batch, target_steps) -> (batch, target_steps, d_model)
        x = self.pos(tgt_embeddings)  # (batch, target_steps, d_model)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask)  # (batch, target_steps, d_model)

        # Project decoder states to vocabulary logits.
        logits = self.generator(x)  # (batch, target_steps, d_model) -> (batch, target_steps, vocab_size)
        return logits  # (batch, target_steps, vocab_size)


# Create and run a sample translation batch.
model = Transformer(vocab_size=37000)
src_ids = torch.randint(0, 37000, (2, 16))  # -> (2, 16)
tgt_ids = torch.randint(0, 37000, (2, 16))  # -> (2, 16)

# Build a causal target mask: (16, 16).
mask_values = torch.ones(16, 16)  # -> (16, 16)
mask_values = mask_values * float('-inf')  # (16, 16)
tgt_mask = torch.triu(mask_values, diagonal=1)  # (16, 16)
logits = model(src_ids, tgt_ids, tgt_mask)  # (2, 16), (2, 16), (16, 16) -> (2, 16, 37000)

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
for step in range(3):
    optimizer.zero_grad()
    logits = model(src_ids, tgt_ids, tgt_mask)  # (2, 4), (2, 4), (4, 4) -> (2, 4, 20)
    vocab_size = logits.size(-1)  # (2, 4, 20) -> scalar
    flat_logits = logits.reshape(-1, vocab_size)  # (2, 4, 20) -> (8, 20)
    flat_targets = train_targets.reshape(-1)  # (2, 4) -> (8)
    loss = criterion(flat_logits, flat_targets)  # (8, 20), (8) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
