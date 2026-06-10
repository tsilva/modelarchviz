# Create and run the embedding block: (2, 4) -> (2, 4, 12).
embeddings = BertEmbeddings(vocab_size=20, hidden_size=12, max_position=8)
input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
token_type_ids = torch.zeros((2, 4), dtype=torch.long)  # -> (2, 4)
embedded = embeddings(input_ids, token_type_ids)  # (2, 4), (2, 4) -> (2, 4, 12)

# Create and run one BERT self-attention block: (2, 4, 12) -> (2, 4, 12).
attention = BertSelfAttention(hidden_size=12, num_heads=3)
hidden_states = torch.randn(2, 4, 12)  # -> (2, 4, 12)
attention_mask = torch.zeros((2, 4), dtype=torch.bool)  # -> (2, 4)
attended = attention(hidden_states, attention_mask)  # (2, 4, 12), (2, 4) -> (2, 4, 12)

# Create and run one encoder layer: (2, 4, 12) -> (2, 4, 12).
layer = BertLayer(hidden_size=12, num_heads=3, intermediate_size=24)
hidden_states = torch.randn(2, 4, 12)  # -> (2, 4, 12)
attention_mask = torch.zeros((2, 4), dtype=torch.bool)  # -> (2, 4)
layer_output = layer(hidden_states, attention_mask)  # (2, 4, 12), (2, 4) -> (2, 4, 12)

# Create and run a sample token batch.
model = BERTBase(vocab_size=30522)
input_ids = torch.randint(0, 30522, (2, 16))  # -> (2, 16)
token_type_ids = torch.zeros((2, 16), dtype=torch.long)  # -> (2, 16)
attention_mask = torch.zeros((2, 16), dtype=torch.bool)  # -> (2, 16)
outputs = model(input_ids, token_type_ids, attention_mask)  # (2, 16), (2, 16), (2, 16) -> tuple
mlm_logits = outputs[0]  # tuple -> (2, 16, 30522)
pooled = outputs[1]  # tuple -> (2, 768)


# Train on a tiny masked-token prediction batch.
model = BERTBase(vocab_size=20, hidden_size=12, num_layers=1)
input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
token_type_ids = torch.zeros((2, 4), dtype=torch.long)  # -> (2, 4)
attention_mask = torch.zeros((2, 4), dtype=torch.bool)  # -> (2, 4)
train_targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])  # -> (2, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(input_ids, token_type_ids, attention_mask)  # (2, 4), (2, 4), (2, 4) -> tuple
    mlm_logits = outputs[0]  # tuple -> (2, 4, 20)
    flat_logits = mlm_logits.reshape(-1, mlm_logits.size(-1))  # (2, 4, 20) -> (8, 20)
    flat_targets = train_targets.reshape(-1)  # (2, 4) -> (8)
    loss = criterion(flat_logits, flat_targets)  # (8, 20), (8) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
