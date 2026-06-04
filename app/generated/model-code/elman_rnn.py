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
