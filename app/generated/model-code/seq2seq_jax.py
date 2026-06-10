import jax
import jax.numpy as jnp
from flax import linen as nn

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
