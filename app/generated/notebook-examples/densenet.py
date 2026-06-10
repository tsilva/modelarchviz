dense_layer = DenseLayer(in_channels=6, growth_rate=4)
dense_input = torch.randn(2, 6, 8, 8)  # -> (2, 6, 8, 8)
dense_output = dense_layer(dense_input)  # (2, 6, 8, 8) -> (2, 10, 8, 8)
print(dense_output.shape)

dense_block = DenseBlock(layer_count=3, in_channels=6, growth_rate=4)
block_input = torch.randn(2, 6, 8, 8)  # -> (2, 6, 8, 8)
block_output = dense_block(block_input)  # (2, 6, 8, 8) -> (2, 18, 8, 8)
print(block_output.shape)

transition = Transition(in_channels=18, out_channels=9)
transition_input = torch.randn(2, 18, 8, 8)  # -> (2, 18, 8, 8)
transition_output = transition(transition_input)  # (2, 18, 8, 8) -> (2, 9, 4, 4)
print(transition_output.shape)

# Create and run a compact image batch: (2, 3, 64, 64) -> (2, 10).
model = DenseNet(
    growth_rate=4,
    block_config=(1, 1, 1, 1),
    num_init_features=8,
    num_classes=10,
)
test_input = torch.randn(2, 3, 64, 64)  # -> (2, 3, 64, 64)
logits = model(test_input)  # (2, 3, 64, 64) -> (2, 10)
print(logits.shape)

# Train on a tiny synthetic image batch.
model = DenseNet(
    growth_rate=4,
    block_config=(1, 1, 1, 1),
    num_init_features=8,
    num_classes=2,
)
train_images = torch.zeros(2, 3, 64, 64)  # -> (2, 3, 64, 64)
train_images[0, :, 8:24, 8:24] = 1.0
train_images[1, :, 40:56, 40:56] = 1.0
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_images)  # (2, 3, 64, 64) -> (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
