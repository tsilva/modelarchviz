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


class LSTMCell(nn.Module):
    def __init__(
        self,
        input_size=32,  # Number of features at each time step.
        hidden_size=64  # Width of hidden and cell states.
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

    def forward(self, x):
        # Build initial recurrent state: (batch, hidden_size).
        batch_size = x.size(0)  # (batch, steps, input_size) -> scalar
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)
        c = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)

        # Run the shared LSTM cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        step_count = x.size(1)  # (batch, steps, input_size) -> scalar
        for t in range(step_count):
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            previous_state = (h, c)
            next_state = self.cell(current_input, previous_state)
            h = next_state[0]  # (batch, hidden_size)
            c = next_state[1]  # (batch, hidden_size)
            states.append(h)

        # Project the final hidden state and pack the full state trace.
        logits = self.readout(h)  # (batch, hidden_size) -> (batch, output_size)
        state_trace = torch.stack(states, dim=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        outputs = (logits, state_trace)
        return outputs


# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = LSTMSequence(input_size=32, hidden_size=64, output_size=10)
sequence = torch.randn(2, 8, 32)  # -> (2, 8, 32)
outputs = model(sequence)
logits = outputs[0]  # (2, 10)
states = outputs[1]  # (2, 8, 64)


# Train on two synthetic sequences with opposite labels.
model = LSTMSequence(input_size=3, hidden_size=8, output_size=2)
train_sequences = torch.tensor(
    [
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
    ]
)  # -> (2, 3, 3)
train_targets = torch.tensor([0, 1])  # -> (2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(train_sequences)
    logits = outputs[0]  # (2, 2)
    loss = criterion(logits, train_targets)  # (2, 2), (2) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
