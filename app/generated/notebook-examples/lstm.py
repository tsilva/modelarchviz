# Create and run one LSTM cell step: (2, 32), state -> next state.
cell = LSTMCell(input_size=32, hidden_size=64)
cell_input = torch.randn(2, 32)  # -> (2, 32)
previous_state = (
    torch.zeros(2, 64),
    torch.zeros(2, 64),
)
next_state = cell(cell_input, previous_state)
next_hidden = next_state[0]  # tuple -> (2, 64)
next_cell = next_state[1]  # tuple -> (2, 64)

# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = LSTMSequence(input_size=32, hidden_size=64, output_size=10)
sequence = torch.randn(2, 8, 32)  # -> (2, 8, 32)
outputs = model(sequence)
logits = outputs[0]  # (2, 10)
states = outputs[1]  # (2, 8, 64)


# Train on two synthetic sequences with opposite labels.
model = LSTMSequence(input_size=3, hidden_size=8, output_size=2)
train_sequences = torch.tensor(
    [
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
    ]
)  # -> (2, 3, 3)
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(train_sequences)
    logits = outputs[0]  # (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
