# Create and run the embedding block: (2, 4) -> (2, 4, 12).
embeddings = BertEmbeddings(vocab_size=20, hidden_size=12, max_position=8)
input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
token_type_ids = jnp.zeros((2, 4), dtype=jnp.int32)  # -> (2, 4)
params = embeddings.init(jax.random.PRNGKey(0), input_ids, token_type_ids, train=False)
embedded = embeddings.apply(params, input_ids, token_type_ids, train=False)  # (2, 4), (2, 4) -> (2, 4, 12)

# Create and run one BERT self-attention block: (2, 4, 12) -> (2, 4, 12).
attention = BertSelfAttention(hidden_size=12, num_heads=3)
hidden_states = jnp.ones((2, 4, 12))  # -> (2, 4, 12)
attention_mask = jnp.ones((2, 1, 1, 4), dtype=jnp.bool_)  # -> (2, 1, 1, 4)
params = attention.init(jax.random.PRNGKey(1), hidden_states, attention_mask)
attended = attention.apply(params, hidden_states, attention_mask)  # (2, 4, 12), (2, 1, 1, 4) -> (2, 4, 12)

# Create and run one encoder layer: (2, 4, 12) -> (2, 4, 12).
layer = BertLayer(hidden_size=12, num_heads=3, intermediate_size=24)
hidden_states = jnp.ones((2, 4, 12))  # -> (2, 4, 12)
attention_mask = jnp.ones((2, 1, 1, 4), dtype=jnp.bool_)  # -> (2, 1, 1, 4)
params = layer.init(jax.random.PRNGKey(2), hidden_states, attention_mask, train=False)
layer_output = layer.apply(params, hidden_states, attention_mask, train=False)  # (2, 4, 12), (2, 1, 1, 4) -> (2, 4, 12)

# Create and run a sample token batch.
model = BERTBase(vocab_size=30522)
input_ids = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)
token_type_ids = jnp.zeros((2, 16), dtype=jnp.int32)  # -> (2, 16)
attention_mask = jnp.ones((2, 1, 1, 16), dtype=jnp.bool_)  # -> (2, 1, 1, 16)
params = model.init(jax.random.PRNGKey(0), input_ids, token_type_ids, attention_mask)
outputs = model.apply(params, input_ids, token_type_ids, attention_mask)
mlm_logits = outputs[0]  # tuple -> (2, 16, 30522)
pooled = outputs[1]  # tuple -> (2, 768)


# Train on a tiny masked-token prediction batch.
model = BERTBase(vocab_size=20, hidden_size=12, num_layers=1)
input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
token_type_ids = jnp.zeros((2, 4), dtype=jnp.int32)  # -> (2, 4)
attention_mask = jnp.ones((2, 1, 1, 4), dtype=jnp.bool_)  # -> (2, 1, 1, 4)
train_targets = jnp.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=jnp.int32)  # -> (2, 4)
params = model.init(jax.random.PRNGKey(1), input_ids, token_type_ids, attention_mask, train=False)


def train_step(params, input_ids, token_type_ids, attention_mask, targets, learning_rate=0.1):
    def loss_fn(current_params):
        outputs = model.apply(current_params, input_ids, token_type_ids, attention_mask, train=False)  # (2, 4), (2, 4), (2, 1, 1, 4) -> tuple
        mlm_logits = outputs[0]  # tuple -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, mlm_logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(mlm_logits, axis=-1)  # (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, input_ids, token_type_ids, attention_mask, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
