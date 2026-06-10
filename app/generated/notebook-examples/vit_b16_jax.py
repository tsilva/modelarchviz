# Create and run patch embedding: (2, 32, 32, 3) -> (2, 4, 24).
patch_embed = PatchEmbed(embed_dim=24, patch_size=16)
images = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
params = patch_embed.init(jax.random.PRNGKey(0), images)
patch_tokens = patch_embed.apply(params, images)  # (2, 32, 32, 3) -> (2, 4, 24)

# Create and run vision self-attention: (2, 5, 24) -> (2, 5, 24).
attention = MultiHeadSelfAttention(embed_dim=24, num_heads=4)
tokens = jnp.ones((2, 5, 24))  # -> (2, 5, 24)
params = attention.init(jax.random.PRNGKey(1), tokens)
attended = attention.apply(params, tokens)  # (2, 5, 24) -> (2, 5, 24)

# Create and run one ViT encoder block: (2, 5, 24) -> (2, 5, 24).
block = EncoderBlock(embed_dim=24, num_heads=4, mlp_dim=48)
tokens = jnp.ones((2, 5, 24))  # -> (2, 5, 24)
params = block.init(jax.random.PRNGKey(2), tokens)
encoded_tokens = block.apply(params, tokens)  # (2, 5, 24) -> (2, 5, 24)

# Create and run a sample image batch: (2, 224, 224, 3) -> (2, 1000).
model = VisionTransformer(num_classes=1000)
test_input = jnp.ones((2, 224, 224, 3))  # -> (2, 224, 224, 3)
params = model.init(jax.random.PRNGKey(0), test_input)
logits = model.apply(params, test_input)  # (2, 224, 224, 3) -> (2, 1000)


# Train on a tiny synthetic image batch.
model = VisionTransformer(num_classes=2, embed_dim=48, depth=1, num_heads=4)
train_images = jnp.zeros((2, 224, 224, 3))  # -> (2, 224, 224, 3)
train_images = train_images.at[0, 32:96, 32:96, :].set(1.0)  # (2, 224, 224, 3)
train_images = train_images.at[1, 128:192, 128:192, :].set(1.0)  # (2, 224, 224, 3)
train_targets = jnp.array([0, 1])  # -> (2)
params = model.init(jax.random.PRNGKey(1), train_images)


def train_step(params, inputs, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs)  # (2, 224, 224, 3) -> (2, 2)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2) -> (2, 2)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 2)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 2), (2, 2) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, train_images, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
