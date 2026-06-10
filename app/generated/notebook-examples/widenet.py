# Create and run one widened residual block: (2, 16, 32, 32) -> (2, 32, 16, 16).
block = WideBasicBlock(in_channels=16, out_channels=32, stride=2, dropout_rate=0.0)
block_input = torch.randn(2, 16, 32, 32)  # -> (2, 16, 32, 32)
block_output = block(block_input)  # (2, 16, 32, 32) -> (2, 32, 16, 16)

# Create and run a sample CIFAR-size image batch: (2, 3, 32, 32) -> (2, 10).
model = WideNet(depth=28, widen_factor=10, dropout_rate=0.0, num_classes=10)
test_input = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
logits = model(test_input)  # (2, 3, 32, 32) -> (2, 10)


# Train on a tiny synthetic CIFAR-size batch.
model = WideNet(depth=10, widen_factor=1, dropout_rate=0.0, num_classes=2)
train_images = torch.zeros(2, 3, 32, 32)  # -> (2, 3, 32, 32)
train_images[0, :, 4:16, 4:16] = 1.0
train_images[1, :, 16:28, 16:28] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 32, 32) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
