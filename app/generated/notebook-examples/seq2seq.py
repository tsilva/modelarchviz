# Create and run one LSTM cell step: (2, 128), two (2, 256) states -> two (2, 256) states.
cell = LSTMCell(input_size=128, hidden_size=256)
cell_input = torch.randn(2, 128)  # -> (2, 128)
previous_state = (
    torch.zeros(2, 256),
    torch.zeros(2, 256),
)
next_state = cell(cell_input, previous_state)
next_h = next_state[0]  # (2, 256)
next_c = next_state[1]  # (2, 256)

# Create and run a sample encoder: (2, 7) -> context and encoder trace.
encoder = Seq2SeqEncoder(vocab_size=32000, embedding_size=128, hidden_size=256)
source_ids = torch.randint(0, 32000, (2, 7))  # -> (2, 7)
encoder_outputs = encoder(source_ids)
context = encoder_outputs[0]
encoder_trace = encoder_outputs[1]  # (2, 7, 256)

# Create and run a sample decoder from an encoder context.
decoder = Seq2SeqDecoder(vocab_size=32000, embedding_size=128, hidden_size=256)
decoder_input_ids = torch.randint(0, 32000, (2, 6))  # -> (2, 6)
decoder_outputs = decoder(decoder_input_ids, context)
decoder_logits = decoder_outputs[0]  # (2, 6, 32000)
decoder_trace = decoder_outputs[1]  # (2, 6, 256)

# Create and run a toy source-to-target batch.
model = Seq2Seq(source_vocab_size=32000, target_vocab_size=32000)
source_ids = torch.randint(0, 32000, (2, 7))  # -> (2, 7)
decoder_input_ids = torch.randint(0, 32000, (2, 6))  # -> (2, 6)
outputs = model(source_ids, decoder_input_ids)
logits = outputs[0]  # (2, 6, 32000)
encoder_trace = outputs[1]  # (2, 7, 256)
decoder_trace = outputs[2]  # (2, 6, 256)

# Train on two tiny symbolic transductions with teacher forcing.
model = Seq2Seq(source_vocab_size=12, target_vocab_size=12, embedding_size=16, hidden_size=32)
source_ids = torch.tensor(
    [
        [3, 4, 5, 0],
        [6, 7, 8, 0],
    ]
)  # -> (2, 4)
decoder_input_ids = torch.tensor(
    [
        [1, 5, 4],
        [1, 8, 7],
    ]
)  # -> (2, 3)
target_ids = torch.tensor(
    [
        [5, 4, 2],
        [8, 7, 2],
    ]
)  # -> (2, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny sequence pairs.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(source_ids, decoder_input_ids)
    logits = outputs[0]  # (2, 3, 12)
    flat_logits = logits.reshape(-1, logits.size(-1))  # (2, 3, 12) -> (6, 12)
    flat_targets = target_ids.reshape(-1)  # (2, 3) -> (6)
    loss = criterion(flat_logits, flat_targets)
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
