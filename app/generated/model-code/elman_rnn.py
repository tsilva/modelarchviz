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
        batch_size = x.size(0)
        hidden_shape = (batch_size, self.hidden_size)
        h = torch.zeros(hidden_shape, device=x.device)

        # Run the shared recurrent cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        step_count = x.size(1)
        for t in range(step_count):
            current_input = x[:, t]
            input_hidden = self.input_to_hidden(current_input)
            recurrent_hidden = self.hidden_to_hidden(h)
            hidden_sum = input_hidden + recurrent_hidden
            h = torch.tanh(hidden_sum)
            states.append(h)

        # Project the final state and pack the full state trace.
        logits = self.hidden_to_output(h)
        state_trace = torch.stack(states, dim=1)
        outputs = (logits, state_trace)
        return outputs


# Create and run a sample sequence: (2, 8, 32) -> logits and states.
model = ElmanRNN(input_size=32, hidden_size=64, output_size=10)
sequence = torch.randn(2, 8, 32)
outputs = model(sequence)
logits = outputs[0]
states = outputs[1]

# logits: (2, 10), states: (2, 8, 64)
