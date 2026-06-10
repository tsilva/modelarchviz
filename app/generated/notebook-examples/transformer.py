position_encoder = PositionalEncoding(d_model=8, max_len=16)
token_embeddings = torch.zeros(2, 4, 8)  # -> (2, 4, 8)
position_encoded = position_encoder(token_embeddings)  # (2, 4, 8)
print(position_encoded.shape)

attention = MultiHeadAttention(d_model=8, nhead=2)
query = torch.randn(2, 3, 8)  # -> (2, 3, 8)
key = torch.randn(2, 4, 8)  # -> (2, 4, 8)
value = torch.randn(2, 4, 8)  # -> (2, 4, 8)
attended = attention(query, key, value)  # (2, 3, 8)
print(attended.shape)

encoder_layer = EncoderLayer(d_model=8, nhead=2, d_ff=32)
encoder_input = torch.randn(2, 4, 8)  # -> (2, 4, 8)
encoder_output = encoder_layer(encoder_input)  # (2, 4, 8)
print(encoder_output.shape)

decoder_layer = DecoderLayer(d_model=8, nhead=2, d_ff=32)
decoder_input = torch.randn(2, 4, 8)  # -> (2, 4, 8)
encoder_memory = torch.randn(2, 5, 8)  # -> (2, 5, 8)
mask_values = torch.ones(4, 4)  # -> (4, 4)
mask_values = mask_values * float('-inf')  # (4, 4)
target_mask = torch.triu(mask_values, diagonal=1)  # (4, 4)
decoder_output = decoder_layer(decoder_input, encoder_memory, target_mask)  # (2, 4, 8)
print(decoder_output.shape)

# Create and run a sample translation batch.
model = Transformer(vocab_size=37000)
src_ids = torch.randint(0, 37000, (2, 16))  # -> (2, 16)
tgt_ids = torch.randint(0, 37000, (2, 16))  # -> (2, 16)

# Build a causal target mask: (16, 16).
mask_values = torch.ones(16, 16)  # -> (16, 16)
mask_values = mask_values * float('-inf')  # (16, 16)
tgt_mask = torch.triu(mask_values, diagonal=1)  # (16, 16)
logits = model(src_ids, tgt_ids, tgt_mask)  # (2, 16), (2, 16), (16, 16) -> (2, 16, 37000)
print(logits.shape)

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
