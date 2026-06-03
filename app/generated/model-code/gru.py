import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(
        self,
        input_size=32,  # Number of features at each time step.
        hidden_size=64  # Width of the recurrent hidden state.
    ):
        super().__init__()

        # Register paired input and recurrent projections for each GRU gate.
        self.x_z = nn.Linear(input_size, hidden_size)
        self.h_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x_r = nn.Linear(input_size, hidden_size)
        self.h_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x_n = nn.Linear(input_size, hidden_size)
        self.h_n = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        # Compute update gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_z = self.x_z(x)
        h_z = self.h_z(h)
        z_pre = x_z + h_z
        z = torch.sigmoid(z_pre)

        # Compute reset gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_r = self.x_r(x)
        h_r = self.h_r(h)
        r_pre = x_r + h_r
        r = torch.sigmoid(r_pre)

        # Compute candidate state from reset hidden state: (batch, hidden_size).
        reset_h = r * h
        x_n = self.x_n(x)
        h_n = self.h_n(reset_h)
        n_pre = x_n + h_n
        n = torch.tanh(n_pre)

        # Blend previous and candidate states: (batch, hidden_size).
        keep_h = z * h
        candidate_h = (1 - z) * n
        h_next = candidate_h + keep_h
        return h_next


class GRUSequence(nn.Module):
    def __init__(
        self,
        input_size=32,  # Number of features at each time step.
        hidden_size=64,  # Width of the recurrent hidden state.
        output_size=10  # Number of output classes.
    ):
        super().__init__()

        # Register the shared recurrent cell and final readout projection.
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        batch_size = x.size(0)
        hidden_shape = (batch_size, self.hidden_size)
        h = torch.zeros(hidden_shape, device=x.device)

        # Run the shared GRU cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        step_count = x.size(1)
        for t in range(step_count):
            current_input = x[:, t]
            h = self.cell(current_input, h)
            states.append(h)

        # Project the final hidden state and pack the full state trace.
        logits = self.readout(h)
        state_trace = torch.stack(states, dim=1)
        outputs = (logits, state_trace)
        return outputs


# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = GRUSequence(input_size=32, hidden_size=64, output_size=10)
sequence = torch.randn(2, 8, 32)
outputs = model(sequence)
logits = outputs[0]
states = outputs[1]


# Train on two synthetic sequences with opposite labels.
model = GRUSequence(input_size=3, hidden_size=8, output_size=2)
train_sequences = torch.tensor(
    [
        [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0]],
    ]
)
train_targets = torch.tensor([0, 1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    outputs = model(train_sequences)
    logits = outputs[0]
    loss = criterion(logits, train_targets)
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()
