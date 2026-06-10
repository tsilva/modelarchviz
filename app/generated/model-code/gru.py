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
        x_z = self.x_z(x)  # (batch, input_size) -> (batch, hidden_size)
        h_z = self.h_z(h)  # (batch, hidden_size)
        z_pre = x_z + h_z  # (batch, hidden_size)
        z = torch.sigmoid(z_pre)  # (batch, hidden_size)

        # Compute reset gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        x_r = self.x_r(x)  # (batch, input_size) -> (batch, hidden_size)
        h_r = self.h_r(h)  # (batch, hidden_size)
        r_pre = x_r + h_r  # (batch, hidden_size)
        r = torch.sigmoid(r_pre)  # (batch, hidden_size)

        # Compute candidate state from reset hidden state: (batch, hidden_size).
        reset_h = r * h  # (batch, hidden_size)
        x_n = self.x_n(x)  # (batch, input_size) -> (batch, hidden_size)
        h_n = self.h_n(reset_h)  # (batch, hidden_size)
        n_pre = x_n + h_n  # (batch, hidden_size)
        n = torch.tanh(n_pre)  # (batch, hidden_size)

        # Blend previous and candidate states: (batch, hidden_size).
        keep_h = z * h  # (batch, hidden_size)
        candidate_h = (1 - z) * n  # (batch, hidden_size)
        h_next = candidate_h + keep_h  # (batch, hidden_size)
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
        batch_size = x.size(0)  # (batch, steps, input_size) -> scalar
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)

        # Run the shared GRU cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        step_count = x.size(1)  # (batch, steps, input_size) -> scalar
        for t in range(step_count):
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            h = self.cell(current_input, h)  # (batch, input_size), (batch, hidden_size) -> (batch, hidden_size)
            states.append(h)

        # Project the final hidden state and pack the full state trace.
        logits = self.readout(h)  # (batch, hidden_size) -> (batch, output_size)
        state_trace = torch.stack(states, dim=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        outputs = (logits, state_trace)
        return outputs

# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = GRUSequence(input_size=32, hidden_size=64, output_size=10)
sequence = torch.randn(2, 8, 32)  # -> (2, 8, 32)
outputs = model(sequence)
logits = outputs[0]  # (2, 10)
states = outputs[1]  # (2, 8, 64)

# Train the same model on two synthetic sequences with opposite labels.
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
