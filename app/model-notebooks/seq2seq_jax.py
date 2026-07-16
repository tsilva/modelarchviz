# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class LSTMCell(nn.Module):
    hidden_size: int = 256

    @nn.compact
    def __call__(self, x, state):
        # Unpack recurrent state: tuple -> two (batch, hidden_size) tensors.
        h, c = state  # ((batch, hidden_size), (batch, hidden_size))

        # Compute input gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_i = nn.Dense(self.hidden_size, name='x_i')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_i = nn.Dense(self.hidden_size, use_bias=False, name='h_i')(h)  # (batch, hidden_size)
        i_pre = x_i + h_i  # (batch, hidden_size)
        i = nn.sigmoid(i_pre)  # (batch, hidden_size)

        # Compute forget gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_f = nn.Dense(self.hidden_size, name='x_f')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_f = nn.Dense(self.hidden_size, use_bias=False, name='h_f')(h)  # (batch, hidden_size)
        f_pre = x_f + h_f  # (batch, hidden_size)
        f = nn.sigmoid(f_pre)  # (batch, hidden_size)

        # Compute candidate memory: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_g = nn.Dense(self.hidden_size, name='x_g')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_g = nn.Dense(self.hidden_size, use_bias=False, name='h_g')(h)  # (batch, hidden_size)
        g_pre = x_g + h_g  # (batch, hidden_size)
        g = jnp.tanh(g_pre)  # (batch, hidden_size)

        # Compute output gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_o = nn.Dense(self.hidden_size, name='x_o')(x)  # (batch, input_size) -> (batch, hidden_size)
        h_o = nn.Dense(self.hidden_size, use_bias=False, name='h_o')(h)  # (batch, hidden_size)
        o_pre = x_o + h_o  # (batch, hidden_size)
        o = nn.sigmoid(o_pre)  # (batch, hidden_size)

        # Blend previous memory with candidate memory: (batch, hidden_size).
        forget_c = f * c  # (batch, hidden_size)
        write_c = i * g  # (batch, hidden_size)
        c_next = forget_c + write_c  # (batch, hidden_size)

        # Read hidden state from updated memory: (batch, hidden_size).
        c_readout = jnp.tanh(c_next)  # (batch, hidden_size)
        h_next = o * c_readout  # (batch, hidden_size)
        next_state = (h_next, c_next)
        return next_state


# %% [notebook-only]
# Create and run one LSTM cell step: (2, 128), two (2, 256) states -> two (2, 256) states.
example_cell = LSTMCell(hidden_size=256)
example_cell_input = jnp.ones((2, 128))  # -> (2, 128)
example_previous_state = (
    jnp.zeros((2, 256)),
    jnp.zeros((2, 256)),
)
cell_params = example_cell.init(jax.random.PRNGKey(0), example_cell_input, example_previous_state)
example_next_state = example_cell.apply(cell_params, example_cell_input, example_previous_state)
next_h = example_next_state[0]  # (2, 256)
next_c = example_next_state[1]  # (2, 256)
print("next_state shape:", example_next_state.shape)

# %%
class Seq2SeqEncoder(nn.Module):
    vocab_size: int = 32000
    embedding_size: int = 128
    hidden_size: int = 256

    @nn.compact
    def __call__(self, source_ids):
        # Reverse source tokens before encoding, matching the original Seq2Seq optimization trick.
        batch_size = source_ids.shape[0]  # (batch, source_steps) -> scalar
        source_steps = source_ids.shape[1]  # (batch, source_steps) -> scalar
        source_order = jnp.arange(source_steps - 1, -1, -1)  # -> (source_steps)
        reversed_ids = source_ids[:, source_order]  # (batch, source_steps)
        embeddings = nn.Embed(self.vocab_size, self.embedding_size, name='source_embed')(reversed_ids)

        # Build the initial encoder state: two (batch, hidden_size) tensors.
        state_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = jnp.zeros(state_shape)  # -> (batch, hidden_size)
        c = jnp.zeros(state_shape)  # -> (batch, hidden_size)

        # Compress the whole source sequence into the final recurrent state.
        encoder_states = []
        cell = LSTMCell(self.hidden_size, name='encoder_cell')
        for t in range(source_steps):
            current_embedding = embeddings[:, t]  # (batch, source_steps, embedding_size) -> (batch, embedding_size)
            state = cell(current_embedding, (h, c))
            h = state[0]  # (batch, hidden_size)
            c = state[1]  # (batch, hidden_size)
            encoder_states.append(h)

        # Return the context state and a trace for inspection.
        context = (h, c)
        encoder_trace = jnp.stack(encoder_states, axis=1)  # list of (batch, hidden_size) -> (batch, source_steps, hidden_size)
        outputs = (context, encoder_trace)
        return outputs


# %% [notebook-only]
# Create and run a sample encoder: (2, 7) -> context and encoder trace.
encoder = Seq2SeqEncoder(vocab_size=32000, embedding_size=128, hidden_size=256)
source_ids = jnp.ones((2, 7), dtype=jnp.int32)  # -> (2, 7)
encoder_params = encoder.init(jax.random.PRNGKey(1), source_ids)
encoder_outputs = encoder.apply(encoder_params, source_ids)
context = encoder_outputs[0]
encoder_trace = encoder_outputs[1]  # (2, 7, 256)
print("encoder_trace shape:", encoder_trace.shape)

# %%
class Seq2SeqDecoder(nn.Module):
    vocab_size: int = 32000
    embedding_size: int = 128
    hidden_size: int = 256

    @nn.compact
    def __call__(self, decoder_input_ids, context):
        # Teacher-force shifted target tokens through the decoder.
        h, c = context  # ((batch, hidden_size), (batch, hidden_size))
        target_steps = decoder_input_ids.shape[1]  # (batch, target_steps) -> scalar
        embeddings = nn.Embed(self.vocab_size, self.embedding_size, name='target_embed')(decoder_input_ids)

        # Decode one target position at a time from the encoder context.
        logits_per_step = []
        decoder_states = []
        cell = LSTMCell(self.hidden_size, name='decoder_cell')
        output_projection = nn.Dense(self.vocab_size, name='output_projection')
        for t in range(target_steps):
            current_embedding = embeddings[:, t]  # (batch, target_steps, embedding_size) -> (batch, embedding_size)
            state = cell(current_embedding, (h, c))
            h = state[0]  # (batch, hidden_size)
            c = state[1]  # (batch, hidden_size)
            logits = output_projection(h)  # (batch, hidden_size) -> (batch, vocab_size)
            logits_per_step.append(logits)
            decoder_states.append(h)

        # Pack per-step vocabulary scores and hidden states.
        output_logits = jnp.stack(logits_per_step, axis=1)  # list of (batch, vocab_size) -> (batch, target_steps, vocab_size)
        decoder_trace = jnp.stack(decoder_states, axis=1)  # list of (batch, hidden_size) -> (batch, target_steps, hidden_size)
        outputs = (output_logits, decoder_trace)
        return outputs


# %% [notebook-only]
# Create and run a sample decoder from an encoder context.
decoder = Seq2SeqDecoder(vocab_size=32000, embedding_size=128, hidden_size=256)
decoder_input_ids = jnp.ones((2, 6), dtype=jnp.int32)  # -> (2, 6)
decoder_params = decoder.init(jax.random.PRNGKey(2), decoder_input_ids, context)
decoder_outputs = decoder.apply(decoder_params, decoder_input_ids, context)
decoder_logits = decoder_outputs[0]  # (2, 6, 32000)
decoder_trace = decoder_outputs[1]  # (2, 6, 256)
print("decoder_trace shape:", decoder_trace.shape)

# %%
class Seq2Seq(nn.Module):
    source_vocab_size: int = 32000
    target_vocab_size: int = 32000
    embedding_size: int = 128
    hidden_size: int = 256

    @nn.compact
    def __call__(self, source_ids, decoder_input_ids):
        # Encode source tokens into a fixed-size context state.
        encoder = Seq2SeqEncoder(
            self.source_vocab_size,
            self.embedding_size,
            self.hidden_size,
            name='encoder',
        )
        encoder_outputs = encoder(source_ids)
        context = encoder_outputs[0]
        encoder_trace = encoder_outputs[1]  # (batch, source_steps, hidden_size)

        # Decode shifted target tokens from that context into vocabulary logits.
        decoder = Seq2SeqDecoder(
            self.target_vocab_size,
            self.embedding_size,
            self.hidden_size,
            name='decoder',
        )
        decoder_outputs = decoder(decoder_input_ids, context)
        logits = decoder_outputs[0]  # (batch, target_steps, target_vocab_size)
        decoder_trace = decoder_outputs[1]  # (batch, target_steps, hidden_size)
        outputs = (logits, encoder_trace, decoder_trace)
        return outputs


# %% [notebook-only]
# Create and run a toy source-to-target batch.
example_model = Seq2Seq(source_vocab_size=32000, target_vocab_size=32000)
source_ids = jnp.ones((2, 7), dtype=jnp.int32)  # -> (2, 7)
decoder_input_ids = jnp.ones((2, 6), dtype=jnp.int32)  # -> (2, 6)
example_params = example_model.init(jax.random.PRNGKey(3), source_ids, decoder_input_ids)
example_outputs = example_model.apply(example_params, source_ids, decoder_input_ids)
example_logits = example_outputs[0]  # (2, 6, 32000)
encoder_trace = example_outputs[1]  # (2, 7, 256)
decoder_trace = example_outputs[2]  # (2, 6, 256)
print("logits shape:", example_logits.shape)

# %%
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
