# Create and run a sample image batch: (2, 3, 227, 227) -> (2, 1000).
model = AlexNet(num_classes=1000)
test_input = torch.randn(2, 3, 227, 227)  # -> (2, 3, 227, 227)
logits = model(test_input)  # (2, 3, 227, 227) -> (2, 1000)


# Train on a tiny synthetic image batch.
model = AlexNet(num_classes=2)
train_images = torch.zeros(2, 3, 227, 227)  # -> (2, 3, 227, 227)
train_images[0, :, 40:100, 40:100] = 1.0
train_images[1, :, 120:180, 120:180] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 227, 227) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
