# Create and run a sample batch: (2, 784) -> (2, 10).
model = MLP(input_dim=784, hidden_dim=128, output_dim=10)
inputs = torch.randn(2, 784)  # -> (2, 784)
logits = model(inputs)  # (2, 784) -> (2, 10)

# Train on a tiny synthetic classification batch.
model = MLP(input_dim=4, hidden_dim=8, output_dim=2)
train_inputs = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
    ]
)  # -> (2, 4)
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_inputs)  # (2, 4) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
