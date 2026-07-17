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
        # @arch lstmcell.self-x_i-nn-linear-input_size-hidden_size:start
        self.x_i = nn.Linear(input_size, hidden_size)
        # @arch lstmcell.self-x_i-nn-linear-input_size-hidden_size:end
        # @arch lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false:start
        self.h_i = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch lstmcell.self-h_i-nn-linear-hidden_size-hidden_size-bias-false:end
        # @arch lstmcell.self-x_f-nn-linear-input_size-hidden_size:start
        self.x_f = nn.Linear(input_size, hidden_size)
        # @arch lstmcell.self-x_f-nn-linear-input_size-hidden_size:end
        # @arch lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false:start
        self.h_f = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch lstmcell.self-h_f-nn-linear-hidden_size-hidden_size-bias-false:end
        # @arch lstmcell.self-x_g-nn-linear-input_size-hidden_size:start
        self.x_g = nn.Linear(input_size, hidden_size)
        # @arch lstmcell.self-x_g-nn-linear-input_size-hidden_size:end
        # @arch lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false:start
        self.h_g = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch lstmcell.self-h_g-nn-linear-hidden_size-hidden_size-bias-false:end
        # @arch lstmcell.self-x_o-nn-linear-input_size-hidden_size:start
        self.x_o = nn.Linear(input_size, hidden_size)
        # @arch lstmcell.self-x_o-nn-linear-input_size-hidden_size:end
        # @arch lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false:start
        self.h_o = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch lstmcell.self-h_o-nn-linear-hidden_size-hidden_size-bias-false:end

    def forward(self, x, state):
        # Unpack recurrent state: tuple -> two (batch, hidden_size) tensors.
        h, c = state  # ((batch, hidden_size), (batch, hidden_size))

        # Compute input gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        # @arch lstmcell.forward.x_i-self-x_i-x:start
        x_i = self.x_i(x)  # (batch, input_size) -> (batch, hidden_size)
        # @arch lstmcell.forward.x_i-self-x_i-x:end
        # @arch lstmcell.forward.h_i-self-h_i-h:start
        h_i = self.h_i(h)  # (batch, hidden_size)
        # @arch lstmcell.forward.h_i-self-h_i-h:end
        # @arch lstmcell.forward.i_pre-x_i-h_i:start
        i_pre = x_i + h_i  # (batch, hidden_size)
        # @arch lstmcell.forward.i_pre-x_i-h_i:end
        # @arch lstmcell.forward.i-torch-sigmoid-i_pre:start
        i = torch.sigmoid(i_pre)  # (batch, hidden_size)
        # @arch lstmcell.forward.i-torch-sigmoid-i_pre:end

        # Compute forget gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        # @arch lstmcell.forward.x_f-self-x_f-x:start
        x_f = self.x_f(x)  # (batch, input_size) -> (batch, hidden_size)
        # @arch lstmcell.forward.x_f-self-x_f-x:end
        # @arch lstmcell.forward.h_f-self-h_f-h:start
        h_f = self.h_f(h)  # (batch, hidden_size)
        # @arch lstmcell.forward.h_f-self-h_f-h:end
        # @arch lstmcell.forward.f_pre-x_f-h_f:start
        f_pre = x_f + h_f  # (batch, hidden_size)
        # @arch lstmcell.forward.f_pre-x_f-h_f:end
        # @arch lstmcell.forward.f-torch-sigmoid-f_pre:start
        f = torch.sigmoid(f_pre)  # (batch, hidden_size)
        # @arch lstmcell.forward.f-torch-sigmoid-f_pre:end

        # Compute candidate memory: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        # @arch lstmcell.forward.x_g-self-x_g-x:start
        x_g = self.x_g(x)  # (batch, input_size) -> (batch, hidden_size)
        # @arch lstmcell.forward.x_g-self-x_g-x:end
        # @arch lstmcell.forward.h_g-self-h_g-h:start
        h_g = self.h_g(h)  # (batch, hidden_size)
        # @arch lstmcell.forward.h_g-self-h_g-h:end
        # @arch lstmcell.forward.g_pre-x_g-h_g:start
        g_pre = x_g + h_g  # (batch, hidden_size)
        # @arch lstmcell.forward.g_pre-x_g-h_g:end
        # @arch lstmcell.forward.g-torch-tanh-g_pre:start
        g = torch.tanh(g_pre)  # (batch, hidden_size)
        # @arch lstmcell.forward.g-torch-tanh-g_pre:end

        # Compute output gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        # @arch lstmcell.forward.x_o-self-x_o-x:start
        x_o = self.x_o(x)  # (batch, input_size) -> (batch, hidden_size)
        # @arch lstmcell.forward.x_o-self-x_o-x:end
        # @arch lstmcell.forward.h_o-self-h_o-h:start
        h_o = self.h_o(h)  # (batch, hidden_size)
        # @arch lstmcell.forward.h_o-self-h_o-h:end
        # @arch lstmcell.forward.o_pre-x_o-h_o:start
        o_pre = x_o + h_o  # (batch, hidden_size)
        # @arch lstmcell.forward.o_pre-x_o-h_o:end
        # @arch lstmcell.forward.o-torch-sigmoid-o_pre:start
        o = torch.sigmoid(o_pre)  # (batch, hidden_size)
        # @arch lstmcell.forward.o-torch-sigmoid-o_pre:end

        # Blend previous memory with candidate memory: (batch, hidden_size).
        # @arch lstmcell.forward.forget_c-f-c:start
        forget_c = f * c  # (batch, hidden_size)
        # @arch lstmcell.forward.forget_c-f-c:end
        # @arch lstmcell.forward.write_c-i-g:start
        write_c = i * g  # (batch, hidden_size)
        # @arch lstmcell.forward.write_c-i-g:end
        # @arch lstmcell.forward.c_next-forget_c-write_c:start
        c_next = forget_c + write_c  # (batch, hidden_size)
        # @arch lstmcell.forward.c_next-forget_c-write_c:end

        # Read hidden state from updated memory: (batch, hidden_size).
        # @arch lstmcell.forward.c_readout-torch-tanh-c_next:start
        c_readout = torch.tanh(c_next)  # (batch, hidden_size)
        # @arch lstmcell.forward.c_readout-torch-tanh-c_next:end
        # @arch lstmcell.forward.h_next-o-c_readout:start
        h_next = o * c_readout  # (batch, hidden_size)
        # @arch lstmcell.forward.h_next-o-c_readout:end
        # @arch lstmcell.forward.next_state-h_next-c_next:start
        next_state = (h_next, c_next)
        # @arch lstmcell.forward.next_state-h_next-c_next:end
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
        # @arch seqnseqencoder.self-embedding-nn-embedding-vocab_size-embedding_size:start
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # @arch seqnseqencoder.self-embedding-nn-embedding-vocab_size-embedding_size:end
        # @arch seqnseqencoder.self-cell-lstmcell-embedding_size-hidden_size:start
        self.cell = LSTMCell(embedding_size, hidden_size)
        # @arch seqnseqencoder.self-cell-lstmcell-embedding_size-hidden_size:end

    def forward(self, source_ids):
        # Reverse source tokens before encoding, matching the original Seq2Seq optimization trick.
        batch_size = source_ids.size(0)  # (batch, source_steps) -> scalar
        source_steps = source_ids.size(1)  # (batch, source_steps) -> scalar
        # @arch seqnseqencoder.forward.source_positions-torch-arange-source_steps-n-n-n-device-source_ids-devic:start
        source_positions = torch.arange(source_steps - 1, -1, -1, device=source_ids.device)  # -> (source_steps)
        # @arch seqnseqencoder.forward.source_positions-torch-arange-source_steps-n-n-n-device-source_ids-devic:end
        # @arch seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions:start
        reversed_ids = source_ids.index_select(1, source_positions)  # (batch, source_steps)
        # @arch seqnseqencoder.forward.reversed_ids-source_ids-index_select-n-source_positions:end
        # @arch seqnseqencoder.forward.embeddings-self-embedding-reversed_ids:start
        embeddings = self.embedding(reversed_ids)  # (batch, source_steps) -> (batch, source_steps, embedding_size)
        # @arch seqnseqencoder.forward.embeddings-self-embedding-reversed_ids:end

        # Build the initial encoder state: two (batch, hidden_size) tensors.
        # @arch seqnseqencoder.forward.state_shape-batch_size-self-hidden_size:start
        state_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        # @arch seqnseqencoder.forward.state_shape-batch_size-self-hidden_size:end
        # @arch seqnseqencoder.forward.h-torch-zeros-state_shape-device-source_ids-device:start
        h = torch.zeros(state_shape, device=source_ids.device)  # -> (batch, hidden_size)
        # @arch seqnseqencoder.forward.h-torch-zeros-state_shape-device-source_ids-device:end
        # @arch seqnseqencoder.forward.c-torch-zeros-state_shape-device-source_ids-device:start
        c = torch.zeros(state_shape, device=source_ids.device)  # -> (batch, hidden_size)
        # @arch seqnseqencoder.forward.c-torch-zeros-state_shape-device-source_ids-device:end

        # Compress the whole source sequence into the final recurrent state.
        encoder_states = []
        # @arch seqnseqencoder.forward.for-t-in-range-source_steps:start
        for t in range(source_steps):
        # @arch seqnseqencoder.forward.for-t-in-range-source_steps:end
            # @arch seqnseqencoder.forward.current_embedding-embeddings-t:start
            current_embedding = embeddings[:, t]  # (batch, source_steps, embedding_size) -> (batch, embedding_size)
            # @arch seqnseqencoder.forward.current_embedding-embeddings-t:end
            # @arch seqnseqencoder.forward.state-self-cell-current_embedding-h-c:start
            state = self.cell(current_embedding, (h, c))
            # @arch seqnseqencoder.forward.state-self-cell-current_embedding-h-c:end
            # @arch seqnseqencoder.forward.h-state-n:start
            h = state[0]  # (batch, hidden_size)
            # @arch seqnseqencoder.forward.h-state-n:end
            # @arch seqnseqencoder.forward.c-state-n:start
            c = state[1]  # (batch, hidden_size)
            # @arch seqnseqencoder.forward.c-state-n:end
            # @arch seqnseqencoder.forward.encoder_states-append-h:start
            encoder_states.append(h)
            # @arch seqnseqencoder.forward.encoder_states-append-h:end

        # Return the context state and a trace for inspection.
        # @arch seqnseqencoder.forward.context-h-c:start
        context = (h, c)
        # @arch seqnseqencoder.forward.context-h-c:end
        # @arch seqnseqencoder.forward.encoder_trace-torch-stack-encoder_states-dim-n:start
        encoder_trace = torch.stack(encoder_states, dim=1)  # list of (batch, hidden_size) -> (batch, source_steps, hidden_size)
        # @arch seqnseqencoder.forward.encoder_trace-torch-stack-encoder_states-dim-n:end
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
        # @arch seqnseqdecoder.self-embedding-nn-embedding-vocab_size-embedding_size:start
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # @arch seqnseqdecoder.self-embedding-nn-embedding-vocab_size-embedding_size:end
        # @arch seqnseqdecoder.self-cell-lstmcell-embedding_size-hidden_size:start
        self.cell = LSTMCell(embedding_size, hidden_size)
        # @arch seqnseqdecoder.self-cell-lstmcell-embedding_size-hidden_size:end
        # @arch seqnseqdecoder.self-output_projection-nn-linear-hidden_size-vocab_size:start
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        # @arch seqnseqdecoder.self-output_projection-nn-linear-hidden_size-vocab_size:end

    def forward(self, decoder_input_ids, context):
        # Teacher-force shifted target tokens through the decoder.
        # @arch seqnseqdecoder.forward.h-c-context:start
        h, c = context  # ((batch, hidden_size), (batch, hidden_size))
        # @arch seqnseqdecoder.forward.h-c-context:end
        target_steps = decoder_input_ids.size(1)  # (batch, target_steps) -> scalar
        # @arch seqnseqdecoder.forward.embeddings-self-embedding-decoder_input_ids:start
        embeddings = self.embedding(decoder_input_ids)  # (batch, target_steps) -> (batch, target_steps, embedding_size)
        # @arch seqnseqdecoder.forward.embeddings-self-embedding-decoder_input_ids:end

        # Decode one target position at a time from the encoder context.
        logits_per_step = []
        decoder_states = []
        # @arch seqnseqdecoder.forward.for-t-in-range-target_steps:start
        for t in range(target_steps):
        # @arch seqnseqdecoder.forward.for-t-in-range-target_steps:end
            # @arch seqnseqdecoder.forward.current_embedding-embeddings-t:start
            current_embedding = embeddings[:, t]  # (batch, target_steps, embedding_size) -> (batch, embedding_size)
            # @arch seqnseqdecoder.forward.current_embedding-embeddings-t:end
            # @arch seqnseqdecoder.forward.state-self-cell-current_embedding-h-c:start
            state = self.cell(current_embedding, (h, c))
            # @arch seqnseqdecoder.forward.state-self-cell-current_embedding-h-c:end
            # @arch seqnseqdecoder.forward.h-state-n:start
            h = state[0]  # (batch, hidden_size)
            # @arch seqnseqdecoder.forward.h-state-n:end
            # @arch seqnseqdecoder.forward.c-state-n:start
            c = state[1]  # (batch, hidden_size)
            # @arch seqnseqdecoder.forward.c-state-n:end
            # @arch seqnseqdecoder.forward.logits-self-output_projection-h:start
            logits = self.output_projection(h)  # (batch, hidden_size) -> (batch, vocab_size)
            # @arch seqnseqdecoder.forward.logits-self-output_projection-h:end
            # @arch seqnseqdecoder.forward.logits_per_step-append-logits:start
            logits_per_step.append(logits)
            # @arch seqnseqdecoder.forward.logits_per_step-append-logits:end
            # @arch seqnseqdecoder.forward.decoder_states-append-h:start
            decoder_states.append(h)
            # @arch seqnseqdecoder.forward.decoder_states-append-h:end

        # Pack per-step vocabulary scores and hidden states.
        # @arch seqnseqdecoder.forward.output_logits-torch-stack-logits_per_step-dim-n:start
        output_logits = torch.stack(logits_per_step, dim=1)  # list of (batch, vocab_size) -> (batch, target_steps, vocab_size)
        # @arch seqnseqdecoder.forward.output_logits-torch-stack-logits_per_step-dim-n:end
        # @arch seqnseqdecoder.forward.decoder_trace-torch-stack-decoder_states-dim-n:start
        decoder_trace = torch.stack(decoder_states, dim=1)  # list of (batch, hidden_size) -> (batch, target_steps, hidden_size)
        # @arch seqnseqdecoder.forward.decoder_trace-torch-stack-decoder_states-dim-n:end
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
        # @arch seqnseq.forward.encoder_outputs-self-encoder-source_ids:start
        encoder_outputs = self.encoder(source_ids)
        # @arch seqnseq.forward.encoder_outputs-self-encoder-source_ids:end
        # @arch seqnseq.forward.context-encoder_outputs-n:start
        context = encoder_outputs[0]
        # @arch seqnseq.forward.context-encoder_outputs-n:end
        encoder_trace = encoder_outputs[1]  # (batch, source_steps, hidden_size)

        # Decode shifted target tokens from that context into vocabulary logits.
        # @arch seqnseq.forward.decoder_outputs-self-decoder-decoder_input_ids-context:start
        decoder_outputs = self.decoder(decoder_input_ids, context)
        # @arch seqnseq.forward.decoder_outputs-self-decoder-decoder_input_ids-context:end
        # @arch seqnseq.forward.logits-decoder_outputs-n:start
        logits = decoder_outputs[0]  # (batch, target_steps, target_vocab_size)
        # @arch seqnseq.forward.logits-decoder_outputs-n:end
        # @arch seqnseq.forward.decoder_trace-decoder_outputs-n:start
        decoder_trace = decoder_outputs[1]  # (batch, target_steps, hidden_size)
        # @arch seqnseq.forward.decoder_trace-decoder_outputs-n:end
        # @arch seqnseq.forward.outputs-logits-encoder_trace-decoder_trace:start
        outputs = (logits, encoder_trace, decoder_trace)
        # @arch seqnseq.forward.outputs-logits-encoder_trace-decoder_trace:end
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
