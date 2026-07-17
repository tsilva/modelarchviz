# %%
import torch
import torch.nn as nn


# %%
class LSTMCell(nn.Module):
    # @arch lstmcell.def-__init__:start
    def __init__(
    # @arch lstmcell.def-__init__:end
        # @arch lstmcell.self:start
        self,
        # @arch lstmcell.self:end
        input_size=32,  # Number of features at each time step.
        hidden_size=64  # Width of hidden and cell states.
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
        # @arch lstmcell.forward.h-c-state:start
        h, c = state  # ((batch, hidden_size), (batch, hidden_size))
        # @arch lstmcell.forward.h-c-state:end

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
        # @arch lstmcell.forward.return-next_state:start
        return next_state
        # @arch lstmcell.forward.return-next_state:end


# %% [notebook-only]
# Create and run one LSTM cell step: (2, 32), state -> next state.
example_cell = LSTMCell(input_size=32, hidden_size=64)
example_input = torch.randn(2, 32)  # -> (2, 32)
example_previous_state = (
    torch.zeros(2, 64),
    torch.zeros(2, 64),
)
example_next_state = example_cell(example_input, example_previous_state)
example_hidden = example_next_state[0]  # tuple -> (2, 64)
example_cell_state = example_next_state[1]  # tuple -> (2, 64)
print("next hidden shape:", example_hidden.shape, "next example_cell shape:", example_cell_state.shape)


# %%
class LSTMSequence(nn.Module):
    def __init__(
        self,
        input_size=32,  # Number of features at each time step.
        hidden_size=64,  # Width of hidden and cell states.
        output_size=10  # Number of output classes.
    ):
        super().__init__()

        # Register the shared recurrent cell and final readout projection.
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)

    # @arch lstmsequence.def-forward-self-x:start
    def forward(self, x):
    # @arch lstmsequence.def-forward-self-x:end
        # Build initial recurrent state: (batch, hidden_size).
        # @arch lstmsequence.forward.batch_size-x-size-n:start
        batch_size = x.size(0)  # (batch, steps, input_size) -> scalar
        # @arch lstmsequence.forward.batch_size-x-size-n:end
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        # @arch lstmsequence.forward.h-torch-zeros-hidden_shape-device-x-device:start
        h = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)
        # @arch lstmsequence.forward.h-torch-zeros-hidden_shape-device-x-device:end
        # @arch lstmsequence.forward.c-torch-zeros-hidden_shape-device-x-device:start
        c = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)
        # @arch lstmsequence.forward.c-torch-zeros-hidden_shape-device-x-device:end

        # Run the shared LSTM cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        step_count = x.size(1)  # (batch, steps, input_size) -> scalar
        # @arch lstmsequence.forward.for-t-in-range-step_count:start
        for t in range(step_count):
        # @arch lstmsequence.forward.for-t-in-range-step_count:end
            # @arch lstmsequence.forward.current_input-x-t:start
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            # @arch lstmsequence.forward.current_input-x-t:end
            # @arch lstmsequence.forward.previous_state-h-c:start
            previous_state = (h, c)
            # @arch lstmsequence.forward.previous_state-h-c:end
            # @arch lstmsequence.forward.next_state-self-cell-current_input-previous_state:start
            next_state = self.cell(current_input, previous_state)
            # @arch lstmsequence.forward.next_state-self-cell-current_input-previous_state:end
            # @arch lstmsequence.forward.h-next_state-n:start
            h = next_state[0]  # (batch, hidden_size)
            # @arch lstmsequence.forward.h-next_state-n:end
            # @arch lstmsequence.forward.c-next_state-n:start
            c = next_state[1]  # (batch, hidden_size)
            # @arch lstmsequence.forward.c-next_state-n:end
            # @arch lstmsequence.forward.states-append-h:start
            states.append(h)
            # @arch lstmsequence.forward.states-append-h:end

        # Project the final hidden state and pack the full state trace.
        logits = self.readout(h)  # (batch, hidden_size) -> (batch, output_size)
        # @arch lstmsequence.forward.state_trace-torch-stack-states-dim-n:start
        state_trace = torch.stack(states, dim=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        # @arch lstmsequence.forward.state_trace-torch-stack-states-dim-n:end
        # @arch lstmsequence.forward.outputs-logits-state_trace:start
        outputs = (logits, state_trace)
        # @arch lstmsequence.forward.outputs-logits-state_trace:end
        # @arch lstmsequence.forward.return-outputs:start
        return outputs
        # @arch lstmsequence.forward.return-outputs:end


# %% [notebook-only]
# Create and run a sample sequence: (2, 8, 32) -> logits and states.
example_model = LSTMSequence(input_size=32, hidden_size=64, output_size=10)
example_sequence = torch.randn(2, 8, 32)  # -> (2, 8, 32)
example_outputs = example_model(example_sequence)
example_logits = example_outputs[0]  # (2, 10)
example_states = example_outputs[1]  # (2, 8, 64)
print("example_logits shape:", example_logits.shape, "example_states shape:", example_states.shape)


# %%
# Train on two synthetic sequences with opposite labels.
model = LSTMSequence(input_size=32, hidden_size=64, output_size=10)
train_sequences = torch.zeros(2, 3, 32)  # -> (2, 3, 32)
train_sequences[0, :, 0] = torch.tensor([1.0, 0.5, 1.0])  # (3)
train_sequences[1, :, 1] = torch.tensor([1.0, 0.5, 1.0])  # (3)
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(train_sequences)
    logits = outputs[0]  # (2, 10)
    loss = criterion(logits, train_targets)  # (2, 10), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
