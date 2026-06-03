import torch
import torch.nn as nn


class ElmanRNN(nn.Module):
    def __init__(
        self,
        input_size=32,  # Number of features at each time step.
        hidden_size=64,  # Width of the recurrent hidden state.
        output_size=10  # Number of output classes.
    ):
        super().__init__()

        # Register recurrent projections and the final readout.
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        batch_size = x.size(0)  # (batch, steps, input_size) -> scalar
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        h = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)

        # Run the shared recurrent cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        step_count = x.size(1)  # (batch, steps, input_size) -> scalar
        for t in range(step_count):
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            input_hidden = self.input_to_hidden(current_input)  # (batch, input_size) -> (batch, hidden_size)
            recurrent_hidden = self.hidden_to_hidden(h)  # (batch, hidden_size)
            hidden_sum = input_hidden + recurrent_hidden  # (batch, hidden_size)
            h = torch.tanh(hidden_sum)  # (batch, hidden_size)
            states.append(h)

        # Project the final state and pack the full state trace.
        logits = self.hidden_to_output(h)  # (batch, hidden_size) -> (batch, output_size)
        state_trace = torch.stack(states, dim=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        outputs = (logits, state_trace)
        return outputs


# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = ElmanRNN(input_size=32, hidden_size=64, output_size=10)
sequence = torch.randn(2, 8, 32)  # -> (2, 8, 32)
outputs = model(sequence)
logits = outputs[0]  # (2, 10)
states = outputs[1]  # (2, 8, 64)


# Train on two synthetic sequences with opposite labels.
model = ElmanRNN(input_size=3, hidden_size=8, output_size=2)
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
