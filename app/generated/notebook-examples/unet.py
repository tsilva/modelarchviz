# Create and run one double convolution: (2, 1, 32, 32) -> (2, 8, 32, 32).
block = DoubleConv(in_channels=1, out_channels=8)
block_input = torch.randn(2, 1, 32, 32)  # -> (2, 1, 32, 32)
block_output = block(block_input)  # (2, 1, 32, 32) -> (2, 8, 32, 32)

# Create and run a sample image batch: (2, 1, 572, 572) -> (2, 2, 572, 572).
model = UNet(num_classes=2)
test_input = torch.randn(2, 1, 572, 572)  # -> (2, 1, 572, 572)
logits = model(test_input)  # (2, 1, 572, 572) -> (2, 2, 572, 572)


# Train on two synthetic segmentation masks.
model = UNet(num_classes=2)
train_images = torch.zeros(2, 1, 64, 64)  # -> (2, 1, 64, 64)
train_images[0, :, 8:32, 8:32] = 1.0
train_images[1, :, 32:56, 32:56] = 1.0
train_targets = torch.zeros(2, 64, 64, dtype=torch.long)  # -> (2, 64, 64)
train_targets[0, 8:32, 8:32] = 1
train_targets[1, 32:56, 32:56] = 1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 1, 64, 64) -> (2, 2, 64, 64)
    loss = criterion(logits, train_targets)  # (2, 2, 64, 64), (2, 64, 64) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
