# Create and run causal self-attention: (2, 4, 24) -> (2, 4, 24).
attention = CausalSelfAttention(n_embd=24, n_head=4)
hidden_states = jnp.ones((2, 4, 24))  # -> (2, 4, 24)
mask = jnp.tril(jnp.ones((4, 4))).reshape(1, 1, 4, 4)  # -> (1, 1, 4, 4)
params = attention.init(jax.random.PRNGKey(0), hidden_states, mask)
attended = attention.apply(params, hidden_states, mask)  # (2, 4, 24), (1, 1, 4, 4) -> (2, 4, 24)

# Create and run the GPT feed-forward block: (2, 4, 24) -> (2, 4, 24).
mlp = MLP(n_embd=24, hidden_dim=48)
hidden_states = jnp.ones((2, 4, 24))  # -> (2, 4, 24)
params = mlp.init(jax.random.PRNGKey(1), hidden_states)
mlp_output = mlp.apply(params, hidden_states)  # (2, 4, 24) -> (2, 4, 24)

# Create and run a sample token batch.
model = GPT2Small(vocab_size=50257)
test_input = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)

# Build a causal attention mask: (1, 1, 16, 16).
mask_values = jnp.ones((16, 16))  # -> (16, 16)
mask = jnp.tril(mask_values)  # (16, 16)
mask = mask.reshape(1, 1, 16, 16)  # (16, 16) -> (1, 1, 16, 16)
params = model.init(jax.random.PRNGKey(0), test_input, mask)
logits = model.apply(params, test_input, mask)  # (2, 16), (1, 1, 16, 16) -> (2, 16, 50257)


# Train on a tiny next-token prediction batch.
model = GPT2Small(vocab_size=20, n_layer=1)
input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
train_targets = jnp.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=jnp.int32)  # -> (2, 4)
mask_values = jnp.ones((4, 4))  # -> (4, 4)
mask = jnp.tril(mask_values)  # (4, 4)
mask = mask.reshape(1, 1, 4, 4)  # (4, 4) -> (1, 1, 4, 4)
params = model.init(jax.random.PRNGKey(1), input_ids, mask)


def train_step(params, inputs, mask, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs, mask)  # (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, input_ids, mask, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
