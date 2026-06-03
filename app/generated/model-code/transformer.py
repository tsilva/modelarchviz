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
        positions = torch.arange(max_len)
        position = positions.unsqueeze(1)
        even_indices = torch.arange(0, d_model, 2)
        log_base = torch.log(torch.tensor(10000.0))
        scale = -log_base / d_model
        div_term = torch.exp(even_indices * scale)
        pe = torch.zeros(max_len, d_model)
        sin_values = torch.sin(position * div_term)
        cos_values = torch.cos(position * div_term)
        pe[:, 0::2] = sin_values
        pe[:, 1::2] = cos_values
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add position encodings to embeddings: (batch, steps, d_model).
        seq_len = x.size(1)
        position_encoding = self.pe[:seq_len]
        encoded = x + position_encoding
        return encoded


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,  # Embedding width.
        nhead=8,  # Number of attention heads.
        d_ff=2048  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register self-attention, feed-forward, and normalization layers.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        # Apply self-attention with residual normalization: (batch, steps, d_model).
        attn, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        attn_residual = x + attn
        x = self.norm1(attn_residual)

        # Apply feed-forward block with residual normalization.
        ffn = self.ffn(x)
        ffn_residual = x + ffn
        out = self.norm2(ffn_residual)
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=512,  # Embedding width.
        nhead=8,  # Number of attention heads.
        d_ff=2048  # Feed-forward hidden width.
    ):
        super().__init__()

        # Register masked attention, cross-attention, feed-forward, and normalization layers.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
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
        masked, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        masked_residual = x + masked
        x = self.norm1(masked_residual)

        # Attend over encoder memory with residual normalization.
        cross, _ = self.cross_attn(x, memory, memory)
        cross_residual = x + cross
        x = self.norm2(cross_residual)

        # Apply feed-forward block with residual normalization.
        ffn = self.ffn(x)
        ffn_residual = x + ffn
        out = self.norm3(ffn_residual)
        return out


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
        src_embeddings = self.src_embed(src_ids)
        memory = self.pos(src_embeddings)
        for layer in self.encoder:
            memory = layer(memory)

        # Embed target tokens and decode against source memory.
        tgt_embeddings = self.tgt_embed(tgt_ids)
        x = self.pos(tgt_embeddings)
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask)

        # Project decoder states to vocabulary logits.
        logits = self.generator(x)
        return logits


# Create and run a sample translation batch.
model = Transformer(vocab_size=37000)
src_ids = torch.randint(0, 37000, (2, 16))
tgt_ids = torch.randint(0, 37000, (2, 16))

# Build a causal target mask: (16, 16).
mask_values = torch.ones(16, 16)
mask_values = mask_values * float('-inf')
tgt_mask = torch.triu(mask_values, diagonal=1)
logits = model(src_ids, tgt_ids, tgt_mask)

# logits: (2, 16, 37000)

# Train on a tiny copy-style token batch.
model = Transformer(vocab_size=20, d_model=16, nhead=4, num_layers=1)
src_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
tgt_ids = torch.tensor([[0, 1, 2, 3], [0, 4, 3, 2]])
train_targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
mask_values = torch.ones(4, 4)
mask_values = mask_values * float('-inf')
tgt_mask = torch.triu(mask_values, diagonal=1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(3):
    optimizer.zero_grad()
    logits = model(src_ids, tgt_ids, tgt_mask)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = train_targets.reshape(-1)
    loss = criterion(flat_logits, flat_targets)
    loss.backward()
    optimizer.step()

final_loss = loss.item()
