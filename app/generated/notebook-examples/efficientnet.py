# Create and run a squeeze-excitation gate: (2, 8, 16, 16) -> (2, 8, 16, 16).
gate = SqueezeExcite(channels=8, squeeze_channels=2)
feature_map = torch.randn(2, 8, 16, 16)  # -> (2, 8, 16, 16)
gated = gate(feature_map)  # (2, 8, 16, 16) -> (2, 8, 16, 16)

# Create and run one mobile inverted bottleneck: (2, 8, 16, 16) -> (2, 8, 16, 16).
block = MBConv(in_channels=8, out_channels=8, expand_ratio=1, stride=1, kernel_size=3)
block_input = torch.randn(2, 8, 16, 16)  # -> (2, 8, 16, 16)
block_output = block(block_input)  # (2, 8, 16, 16) -> (2, 8, 16, 16)

# Create and run a sample ImageNet-size batch: (2, 3, 224, 224) -> (2, 1000).
model = EfficientNet(num_classes=1000)
test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
logits = model(test_input)  # (2, 3, 224, 224) -> (2, 1000)


# Train on a tiny synthetic image batch.
model = EfficientNet(num_classes=2)
train_images = torch.zeros(2, 3, 224, 224)  # -> (2, 3, 224, 224)
train_images[0, :, 32:96, 32:96] = 1.0
train_images[1, :, 128:192, 128:192] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 224, 224) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
