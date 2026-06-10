# Create and run causal self-attention: (2, 4, 24) -> (2, 4, 24).
attention = CausalSelfAttention(n_embd=24, n_head=4)
hidden_states = torch.randn(2, 4, 24)  # -> (2, 4, 24)
mask = torch.tril(torch.ones(4, 4)).view(1, 1, 4, 4)  # -> (1, 1, 4, 4)
attended = attention(hidden_states, mask)  # (2, 4, 24), (1, 1, 4, 4) -> (2, 4, 24)

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
