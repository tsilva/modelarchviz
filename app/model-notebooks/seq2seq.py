# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn


# %%
class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size=128,  # Token embedding width.
        hidden_size=256  # Width of hidden and cell states.
    ):
        super().__init__()

        # Register paired input and recurrent projections for each LSTM gate.
        self.x_i = nn.Linear(input_size, hidden_size)
        self.h_i = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x_f = nn.Linear(input_size, hidden_size)
        self.h_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x_g = nn.Linear(input_size, hidden_size)
        self.h_g = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x_o = nn.Linear(input_size, hidden_size)
        self.h_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, state):
        # Unpack recurrent state: tuple -> two (batch, hidden_size) tensors.
        h, c = state  # ((batch, hidden_size), (batch, hidden_size))

        # Compute input gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_i = self.x_i(x)  # (batch, input_size) -> (batch, hidden_size)
        h_i = self.h_i(h)  # (batch, hidden_size)
        i_pre = x_i + h_i  # (batch, hidden_size)
        i = torch.sigmoid(i_pre)  # (batch, hidden_size)

        # Compute forget gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_f = self.x_f(x)  # (batch, input_size) -> (batch, hidden_size)
        h_f = self.h_f(h)  # (batch, hidden_size)
        f_pre = x_f + h_f  # (batch, hidden_size)
        f = torch.sigmoid(f_pre)  # (batch, hidden_size)

        # Compute candidate memory: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_g = self.x_g(x)  # (batch, input_size) -> (batch, hidden_size)
        h_g = self.h_g(h)  # (batch, hidden_size)
        g_pre = x_g + h_g  # (batch, hidden_size)
        g = torch.tanh(g_pre)  # (batch, hidden_size)

        # Compute output gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_o = self.x_o(x)  # (batch, input_size) -> (batch, hidden_size)
        h_o = self.h_o(h)  # (batch, hidden_size)
        o_pre = x_o + h_o  # (batch, hidden_size)
        o = torch.sigmoid(o_pre)  # (batch, hidden_size)

        # Blend previous memory with candidate memory: (batch, hidden_size).
        forget_c = f * c  # (batch, hidden_size)
        write_c = i * g  # (batch, hidden_size)
        c_next = forget_c + write_c  # (batch, hidden_size)

        # Read hidden state from updated memory: (batch, hidden_size).
        c_readout = torch.tanh(c_next)  # (batch, hidden_size)
        h_next = o * c_readout  # (batch, hidden_size)
        next_state = (h_next, c_next)
        return next_state


# %% [notebook-only]
# Create and run one LSTM cell step: (2, 128), two (2, 256) states -> two (2, 256) states.
example_cell = LSTMCell(input_size=128, hidden_size=256)
example_cell_input = torch.randn(2, 128)  # -> (2, 128)
example_previous_state = (
    torch.zeros(2, 256),
    torch.zeros(2, 256),
)
example_next_state = example_cell(example_cell_input, example_previous_state)
next_h = example_next_state[0]  # (2, 256)
next_c = example_next_state[1]  # (2, 256)
print("next_state shape:", example_next_state.shape)

# %%
class Seq2SeqEncoder(nn.Module):
    def __init__(
        self,
        vocab_size=32000,  # Source vocabulary size.
        embedding_size=128,  # Source token embedding width.
        hidden_size=256  # Width of encoder hidden and cell states.
    ):
        super().__init__()

        # Register source token embeddings and the shared encoder LSTM cell.
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.cell = LSTMCell(embedding_size, hidden_size)

    def forward(self, source_ids):
        # Reverse source tokens before encoding, matching the original Seq2Seq optimization trick.
        batch_size = source_ids.size(0)  # (batch, source_steps) -> scalar
        source_steps = source_ids.size(1)  # (batch, source_steps) -> scalar
        source_positions = torch.arange(source_steps - 1, -1, -1, device=source_ids.device)  # -> (source_steps)
        reversed_ids = source_ids.index_select(1, source_positions)  # (batch, source_steps)
        embeddings = self.embedding(reversed_ids)  # (batch, source_steps) -> (batch, source_steps, embedding_size)

        # Build the initial encoder state: two (batch, hidden_size) tensors.
        state_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = torch.zeros(state_shape, device=source_ids.device)  # -> (batch, hidden_size)
        c = torch.zeros(state_shape, device=source_ids.device)  # -> (batch, hidden_size)

        # Compress the whole source sequence into the final recurrent state.
        encoder_states = []
        for t in range(source_steps):
            current_embedding = embeddings[:, t]  # (batch, source_steps, embedding_size) -> (batch, embedding_size)
            state = self.cell(current_embedding, (h, c))
            h = state[0]  # (batch, hidden_size)
            c = state[1]  # (batch, hidden_size)
            encoder_states.append(h)

        # Return the context state and a trace for inspection.
        context = (h, c)
        encoder_trace = torch.stack(encoder_states, dim=1)  # list of (batch, hidden_size) -> (batch, source_steps, hidden_size)
        outputs = (context, encoder_trace)
        return outputs


# %% [notebook-only]
# Create and run a sample encoder: (2, 7) -> context and encoder trace.
encoder = Seq2SeqEncoder(vocab_size=32000, embedding_size=128, hidden_size=256)
source_ids = torch.randint(0, 32000, (2, 7))  # -> (2, 7)
encoder_outputs = encoder(source_ids)
context = encoder_outputs[0]
encoder_trace = encoder_outputs[1]  # (2, 7, 256)
print("encoder_trace shape:", encoder_trace.shape)

# %%
class Seq2SeqDecoder(nn.Module):
    def __init__(
        self,
        vocab_size=32000,  # Target vocabulary size.
        embedding_size=128,  # Target token embedding width.
        hidden_size=256  # Width of decoder hidden and cell states.
    ):
        super().__init__()

        # Register target embeddings, decoder cell, and vocabulary projection.
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.cell = LSTMCell(embedding_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, decoder_input_ids, context):
        # Teacher-force shifted target tokens through the decoder.
        h, c = context  # ((batch, hidden_size), (batch, hidden_size))
        target_steps = decoder_input_ids.size(1)  # (batch, target_steps) -> scalar
        embeddings = self.embedding(decoder_input_ids)  # (batch, target_steps) -> (batch, target_steps, embedding_size)

        # Decode one target position at a time from the encoder context.
        logits_per_step = []
        decoder_states = []
        for t in range(target_steps):
            current_embedding = embeddings[:, t]  # (batch, target_steps, embedding_size) -> (batch, embedding_size)
            state = self.cell(current_embedding, (h, c))
            h = state[0]  # (batch, hidden_size)
            c = state[1]  # (batch, hidden_size)
            logits = self.output_projection(h)  # (batch, hidden_size) -> (batch, vocab_size)
            logits_per_step.append(logits)
            decoder_states.append(h)

        # Pack per-step vocabulary scores and hidden states.
        output_logits = torch.stack(logits_per_step, dim=1)  # list of (batch, vocab_size) -> (batch, target_steps, vocab_size)
        decoder_trace = torch.stack(decoder_states, dim=1)  # list of (batch, hidden_size) -> (batch, target_steps, hidden_size)
        outputs = (output_logits, decoder_trace)
        return outputs


# %% [notebook-only]
# Create and run a sample decoder from an encoder context.
decoder = Seq2SeqDecoder(vocab_size=32000, embedding_size=128, hidden_size=256)
decoder_input_ids = torch.randint(0, 32000, (2, 6))  # -> (2, 6)
decoder_outputs = decoder(decoder_input_ids, context)
decoder_logits = decoder_outputs[0]  # (2, 6, 32000)
decoder_trace = decoder_outputs[1]  # (2, 6, 256)
print("decoder_trace shape:", decoder_trace.shape)

# %%
class Seq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab_size=32000,  # Source vocabulary size.
        target_vocab_size=32000,  # Target vocabulary size.
        embedding_size=128,  # Shared token embedding width.
        hidden_size=256  # Shared encoder and decoder state width.
    ):
        super().__init__()

        # Register the encoder and decoder halves of the transduction model.
        self.encoder = Seq2SeqEncoder(source_vocab_size, embedding_size, hidden_size)
        self.decoder = Seq2SeqDecoder(target_vocab_size, embedding_size, hidden_size)

    def forward(self, source_ids, decoder_input_ids):
        # Encode source tokens into a fixed-size context state.
        encoder_outputs = self.encoder(source_ids)
        context = encoder_outputs[0]
        encoder_trace = encoder_outputs[1]  # (batch, source_steps, hidden_size)

        # Decode shifted target tokens from that context into vocabulary logits.
        decoder_outputs = self.decoder(decoder_input_ids, context)
        logits = decoder_outputs[0]  # (batch, target_steps, target_vocab_size)
        decoder_trace = decoder_outputs[1]  # (batch, target_steps, hidden_size)
        outputs = (logits, encoder_trace, decoder_trace)
        return outputs


# %% [notebook-only]
# Create and run a toy source-to-target batch.
example_model = Seq2Seq(source_vocab_size=32000, target_vocab_size=32000)
source_ids = torch.randint(0, 32000, (2, 7))  # -> (2, 7)
decoder_input_ids = torch.randint(0, 32000, (2, 6))  # -> (2, 6)
example_outputs = example_model(source_ids, decoder_input_ids)
example_logits = example_outputs[0]  # (2, 6, 32000)
encoder_trace = example_outputs[1]  # (2, 7, 256)
decoder_trace = example_outputs[2]  # (2, 6, 256)
print("logits shape:", example_logits.shape)

# %%
# Train on two tiny symbolic transductions with teacher forcing.
model = Seq2Seq(source_vocab_size=12, target_vocab_size=12, embedding_size=16, hidden_size=32)
source_ids = torch.tensor(
    [
        [3, 4, 5, 0],
        [6, 7, 8, 0],
    ]
)  # -> (2, 4)
decoder_input_ids = torch.tensor(
    [
        [1, 5, 4],
        [1, 8, 7],
    ]
)  # -> (2, 3)
target_ids = torch.tensor(
    [
        [5, 4, 2],
        [8, 7, 2],
    ]
)  # -> (2, 3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny sequence pairs.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(source_ids, decoder_input_ids)
    logits = outputs[0]  # (2, 3, 12)
    flat_logits = logits.reshape(-1, logits.size(-1))  # (2, 3, 12) -> (6, 12)
    flat_targets = target_ids.reshape(-1)  # (2, 3) -> (6)
    loss = criterion(flat_logits, flat_targets)
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
