# Create and run one LSTM cell step: (2, 128), two (2, 256) states -> two (2, 256) states.
cell = LSTMCell(hidden_size=256)
cell_input = jnp.ones((2, 128))  # -> (2, 128)
previous_state = (
    jnp.zeros((2, 256)),
    jnp.zeros((2, 256)),
)
cell_params = cell.init(jax.random.PRNGKey(0), cell_input, previous_state)
next_state = cell.apply(cell_params, cell_input, previous_state)
next_h = next_state[0]  # (2, 256)
next_c = next_state[1]  # (2, 256)

# Create and run a sample encoder: (2, 7) -> context and encoder trace.
encoder = Seq2SeqEncoder(vocab_size=32000, embedding_size=128, hidden_size=256)
source_ids = jnp.ones((2, 7), dtype=jnp.int32)  # -> (2, 7)
encoder_params = encoder.init(jax.random.PRNGKey(1), source_ids)
encoder_outputs = encoder.apply(encoder_params, source_ids)
context = encoder_outputs[0]
encoder_trace = encoder_outputs[1]  # (2, 7, 256)

# Create and run a sample decoder from an encoder context.
decoder = Seq2SeqDecoder(vocab_size=32000, embedding_size=128, hidden_size=256)
decoder_input_ids = jnp.ones((2, 6), dtype=jnp.int32)  # -> (2, 6)
decoder_params = decoder.init(jax.random.PRNGKey(2), decoder_input_ids, context)
decoder_outputs = decoder.apply(decoder_params, decoder_input_ids, context)
decoder_logits = decoder_outputs[0]  # (2, 6, 32000)
decoder_trace = decoder_outputs[1]  # (2, 6, 256)

# Create and run a toy source-to-target batch.
model = Seq2Seq(source_vocab_size=32000, target_vocab_size=32000)
source_ids = jnp.ones((2, 7), dtype=jnp.int32)  # -> (2, 7)
decoder_input_ids = jnp.ones((2, 6), dtype=jnp.int32)  # -> (2, 6)
params = model.init(jax.random.PRNGKey(3), source_ids, decoder_input_ids)
outputs = model.apply(params, source_ids, decoder_input_ids)
logits = outputs[0]  # (2, 6, 32000)
encoder_trace = outputs[1]  # (2, 7, 256)
decoder_trace = outputs[2]  # (2, 6, 256)

# Train on two tiny symbolic transductions with teacher forcing.
model = Seq2Seq(source_vocab_size=12, target_vocab_size=12, embedding_size=16, hidden_size=32)
source_ids = jnp.array(
    [
        [3, 4, 5, 0],
        [6, 7, 8, 0],
    ]
)  # -> (2, 4)
decoder_input_ids = jnp.array(
    [
        [1, 5, 4],
        [1, 8, 7],
    ]
)  # -> (2, 3)
target_ids = jnp.array(
    [
        [5, 4, 2],
        [8, 7, 2],
    ]
)  # -> (2, 3)
params = model.init(jax.random.PRNGKey(4), source_ids, decoder_input_ids)


def train_step(params, inputs, decoder_inputs, targets, learning_rate=0.1):
    def loss_fn(current_params):
        outputs = model.apply(current_params, inputs, decoder_inputs)
        logits = outputs[0]  # (batch, target_steps, target_vocab_size)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (batch, target_steps, target_vocab_size)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (batch, target_steps, target_vocab_size)
        token_losses = jnp.sum(one_hot_targets * log_probs, axis=-1)  # (batch, target_steps)
        loss = -jnp.mean(token_losses)  # (batch, target_steps) -> scalar
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny sequence pairs.
for step in range(3):
    params, loss = train_step(params, source_ids, decoder_input_ids, target_ids)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
