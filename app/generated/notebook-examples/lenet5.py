# Create and run a sample image batch: (2, 1, 32, 32) -> (2, 10).
model = LeNet5()
test_input = torch.randn(2, 1, 32, 32)  # -> (2, 1, 32, 32)
logits = model(test_input)  # (2, 1, 32, 32) -> (2, 10)


# Train on a tiny synthetic image batch.
model = LeNet5()
train_images = torch.zeros(2, 1, 32, 32)  # -> (2, 1, 32, 32)
train_images[0, :, 8:16, 8:16] = 1.0
train_images[1, :, 16:24, 16:24] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 1, 32, 32) -> (2, 10)
    loss = criterion(logits, train_targets)  # (2, 10), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
