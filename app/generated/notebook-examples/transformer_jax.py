# Create and run positional encoding: (2, 4, 8) -> (2, 4, 8).
positioner = PositionalEncoding(d_model=8)
embeddings = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
encoded = positioner.init_with_output(jax.random.PRNGKey(0), embeddings)[0]  # (2, 4, 8)

# Create and run multi-head attention: query and memory -> (2, 3, 8).
attention = MultiHeadAttention(d_model=8, nhead=2)
query = jnp.ones((2, 3, 8))  # -> (2, 3, 8)
key = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
value = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
mask = jnp.ones((1, 1, 3, 4))  # -> (1, 1, 3, 4)
params = attention.init(jax.random.PRNGKey(1), query, key, value, mask)
attended = attention.apply(params, query, key, value, mask)  # inputs -> (2, 3, 8)

# Create and run one encoder layer: (2, 4, 8) -> (2, 4, 8).
encoder_layer = EncoderLayer(d_model=8, nhead=2, d_ff=16)
encoder_input = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
params = encoder_layer.init(jax.random.PRNGKey(2), encoder_input)
encoder_output = encoder_layer.apply(params, encoder_input)  # (2, 4, 8) -> (2, 4, 8)

# Create and run one decoder layer: target and memory -> (2, 4, 8).
decoder_layer = DecoderLayer(d_model=8, nhead=2, d_ff=16)
decoder_input = jnp.ones((2, 4, 8))  # -> (2, 4, 8)
encoder_memory = jnp.ones((2, 5, 8))  # -> (2, 5, 8)
mask = jnp.ones((1, 1, 4, 4))  # -> (1, 1, 4, 4)
params = decoder_layer.init(jax.random.PRNGKey(3), decoder_input, encoder_memory, mask)
decoder_output = decoder_layer.apply(params, decoder_input, encoder_memory, mask)  # inputs -> (2, 4, 8)

# Create and run a sample translation batch.
model = Transformer(vocab_size=37000)
src_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)
tgt_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)

# Build a causal target mask: (1, 1, 16, 16).
mask_values = jnp.ones((1, 1, 16, 16))  # -> (1, 1, 16, 16)
tgt_mask = jnp.tril(mask_values)  # (1, 1, 16, 16)
params = model.init(jax.random.PRNGKey(0), src_ids, tgt_ids, tgt_mask)  # inputs -> parameter tree
logits = model.apply(params, src_ids, tgt_ids, tgt_mask)  # (2, 16), (2, 16), (1, 1, 16, 16) -> (2, 16, 37000)

# Train on a tiny copy-style token batch.
model = Transformer(vocab_size=20, d_model=16, nhead=4, num_layers=1)
src_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
tgt_ids = jnp.array([[0, 1, 2, 3], [0, 4, 3, 2]], dtype=jnp.int32)  # -> (2, 4)
train_targets = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
mask_values = jnp.ones((1, 1, 4, 4))  # -> (1, 1, 4, 4)
tgt_mask = jnp.tril(mask_values)  # (1, 1, 4, 4)
params = model.init(jax.random.PRNGKey(1), src_ids, tgt_ids, tgt_mask)  # inputs -> parameter tree


def train_step(params, src_ids, tgt_ids, targets, mask, learning_rate=0.1):
    def loss_fn(current_params):
        logits = model.apply(current_params, src_ids, tgt_ids, mask)  # (2, 4), (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)  # parameter tree -> scalar, gradient tree
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)  # parameter tree
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, src_ids, tgt_ids, train_targets, tgt_mask)  # parameter tree -> parameter tree, scalar

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
