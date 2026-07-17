# %%
import torch
import torch.nn as nn


# %%
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
        # @arch elmanrnn.input_to_hidden:start
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        # @arch elmanrnn.input_to_hidden:end
        # @arch elmanrnn.self-hidden_to_hidden-nn-linear-hidden_size-hidden_size-bias-false:start
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch elmanrnn.self-hidden_to_hidden-nn-linear-hidden_size-hidden_size-bias-false:end
        # @arch elmanrnn.self-hidden_to_output-nn-linear-hidden_size-output_size:start
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        # @arch elmanrnn.self-hidden_to_output-nn-linear-hidden_size-output_size:end

    def forward(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        batch_size = x.size(0)  # (batch, steps, input_size) -> scalar
        # @arch elmanrnn.forward.hidden_shape-batch_size-self-hidden_size:start
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        # @arch elmanrnn.forward.hidden_shape-batch_size-self-hidden_size:end
        # @arch elmanrnn.forward.h-torch-zeros-hidden_shape-device-x-device:start
        h = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)
        # @arch elmanrnn.forward.h-torch-zeros-hidden_shape-device-x-device:end

        # Run the shared recurrent cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        states = []
        # @arch elmanrnn.forward.step_count-x-size-n:start
        step_count = x.size(1)  # (batch, steps, input_size) -> scalar
        # @arch elmanrnn.forward.step_count-x-size-n:end
        # @arch elmanrnn.forward.for-t-in-range-step_count:start
        for t in range(step_count):
        # @arch elmanrnn.forward.for-t-in-range-step_count:end
            # @arch elmanrnn.forward.current_input-x-t:start
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            # @arch elmanrnn.forward.current_input-x-t:end
            # @arch elmanrnn.forward.input_hidden-self-input_to_hidden-current_input:start
            input_hidden = self.input_to_hidden(current_input)  # (batch, input_size) -> (batch, hidden_size)
            # @arch elmanrnn.forward.input_hidden-self-input_to_hidden-current_input:end
            # @arch elmanrnn.forward.recurrent_hidden-self-hidden_to_hidden-h:start
            recurrent_hidden = self.hidden_to_hidden(h)  # (batch, hidden_size)
            # @arch elmanrnn.forward.recurrent_hidden-self-hidden_to_hidden-h:end
            # @arch elmanrnn.forward.hidden_sum-input_hidden-recurrent_hidden:start
            hidden_sum = input_hidden + recurrent_hidden  # (batch, hidden_size)
            # @arch elmanrnn.forward.hidden_sum-input_hidden-recurrent_hidden:end
            # @arch elmanrnn.forward.h-torch-tanh-hidden_sum:start
            h = torch.tanh(hidden_sum)  # (batch, hidden_size)
            # @arch elmanrnn.forward.h-torch-tanh-hidden_sum:end
            # @arch elmanrnn.forward.states-append-h:start
            states.append(h)
            # @arch elmanrnn.forward.states-append-h:end

        # Project the final state and pack the full state trace.
        # @arch elmanrnn.forward.logits-self-hidden_to_output-h:start
        logits = self.hidden_to_output(h)  # (batch, hidden_size) -> (batch, output_size)
        # @arch elmanrnn.forward.logits-self-hidden_to_output-h:end
        # @arch elmanrnn.forward.state_trace-torch-stack-states-dim-n:start
        state_trace = torch.stack(states, dim=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        # @arch elmanrnn.forward.state_trace-torch-stack-states-dim-n:end
        # @arch elmanrnn.forward.outputs-logits-state_trace:start
        outputs = (logits, state_trace)
        # @arch elmanrnn.forward.outputs-logits-state_trace:end
        # @arch elmanrnn.forward.return-outputs:start
        return outputs
        # @arch elmanrnn.forward.return-outputs:end


# %% [notebook-only]
# Create and run a sample sequence: (2, 8, 32) -> logits and states.
example_model = ElmanRNN(input_size=32, hidden_size=64, output_size=10)
example_sequence = torch.randn(2, 8, 32)  # -> (2, 8, 32)
example_outputs = example_model(example_sequence)
example_logits = example_outputs[0]  # (2, 10)
example_states = example_outputs[1]  # (2, 8, 64)
print("logits shape:", example_logits.shape)

# %%
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
