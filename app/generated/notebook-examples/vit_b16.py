# Create and run patch embedding: (2, 3, 32, 32) -> (2, 4, 24).
patch_embed = PatchEmbed(in_channels=3, embed_dim=24, patch_size=16, image_size=32)
images = torch.randn(2, 3, 32, 32)  # -> (2, 3, 32, 32)
patch_tokens = patch_embed(images)  # (2, 3, 32, 32) -> (2, 4, 24)

# Create and run vision self-attention: (2, 5, 24) -> (2, 5, 24).
attention = MultiHeadSelfAttention(embed_dim=24, num_heads=4)
tokens = torch.randn(2, 5, 24)  # -> (2, 5, 24)
attended = attention(tokens)  # (2, 5, 24) -> (2, 5, 24)

# Create and run one ViT encoder block: (2, 5, 24) -> (2, 5, 24).
block = EncoderBlock(embed_dim=24, num_heads=4, mlp_dim=48)
tokens = torch.randn(2, 5, 24)  # -> (2, 5, 24)
encoded_tokens = block(tokens)  # (2, 5, 24) -> (2, 5, 24)

# Create and run a sample image batch: (2, 3, 224, 224) -> (2, 1000).
model = VisionTransformer(num_classes=1000)
test_input = torch.randn(2, 3, 224, 224)  # -> (2, 3, 224, 224)
logits = model(test_input)  # (2, 3, 224, 224) -> (2, 1000)


# Train on a tiny synthetic image batch.
model = VisionTransformer(num_classes=2, embed_dim=48, depth=1, num_heads=4)
train_images = torch.zeros(2, 3, 224, 224)  # -> (2, 3, 224, 224)
train_images[0, :, 32:96, 32:96] = 1.0  # (2, 3, 224, 224)
train_images[1, :, 128:192, 128:192] = 1.0  # (2, 3, 224, 224)
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
