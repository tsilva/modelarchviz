# %%
import torch
import torch.nn as nn


# %%
class GRUCell(nn.Module):
    # @arch grucell.def-__init__:start
    def __init__(
    # @arch grucell.def-__init__:end
        # @arch grucell.self:start
        self,
        # @arch grucell.self:end
        input_size=32,  # Number of features at each time step.
        hidden_size=64  # Width of the recurrent hidden state.
    ):
        super().__init__()

        # Register paired input and recurrent projections for each GRU gate.
        # @arch grucell.self-x_z-nn-linear-input_size-hidden_size:start
        self.x_z = nn.Linear(input_size, hidden_size)
        # @arch grucell.self-x_z-nn-linear-input_size-hidden_size:end
        # @arch grucell.self-h_z-nn-linear-hidden_size-hidden_size-bias-false:start
        self.h_z = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch grucell.self-h_z-nn-linear-hidden_size-hidden_size-bias-false:end
        # @arch grucell.self-x_r-nn-linear-input_size-hidden_size:start
        self.x_r = nn.Linear(input_size, hidden_size)
        # @arch grucell.self-x_r-nn-linear-input_size-hidden_size:end
        # @arch grucell.self-h_r-nn-linear-hidden_size-hidden_size-bias-false:start
        self.h_r = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch grucell.self-h_r-nn-linear-hidden_size-hidden_size-bias-false:end
        # @arch grucell.self-x_n-nn-linear-input_size-hidden_size:start
        self.x_n = nn.Linear(input_size, hidden_size)
        # @arch grucell.self-x_n-nn-linear-input_size-hidden_size:end
        # @arch grucell.self-h_n-nn-linear-hidden_size-hidden_size-bias-false:start
        self.h_n = nn.Linear(hidden_size, hidden_size, bias=False)
        # @arch grucell.self-h_n-nn-linear-hidden_size-hidden_size-bias-false:end

    # @arch grucell.def-forward-self-x-h:start
    def forward(self, x, h):
    # @arch grucell.def-forward-self-x-h:end
        # Compute update gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        # @arch grucell.forward.x_z-self-x_z-x:start
        x_z = self.x_z(x)  # (batch, input_size) -> (batch, hidden_size)
        # @arch grucell.forward.x_z-self-x_z-x:end
        # @arch grucell.forward.h_z-self-h_z-h:start
        h_z = self.h_z(h)  # (batch, hidden_size)
        # @arch grucell.forward.h_z-self-h_z-h:end
        # @arch grucell.forward.z_pre-x_z-h_z:start
        z_pre = x_z + h_z  # (batch, hidden_size)
        # @arch grucell.forward.z_pre-x_z-h_z:end
        # @arch grucell.forward.z-torch-sigmoid-z_pre:start
        z = torch.sigmoid(z_pre)  # (batch, hidden_size)
        # @arch grucell.forward.z-torch-sigmoid-z_pre:end

        # Compute reset gate: (batch, input_size) + (batch, hidden_size) -> (batch, hidden_size).
        # @arch grucell.forward.x_r-self-x_r-x:start
        x_r = self.x_r(x)  # (batch, input_size) -> (batch, hidden_size)
        # @arch grucell.forward.x_r-self-x_r-x:end
        # @arch grucell.forward.h_r-self-h_r-h:start
        h_r = self.h_r(h)  # (batch, hidden_size)
        # @arch grucell.forward.h_r-self-h_r-h:end
        # @arch grucell.forward.r_pre-x_r-h_r:start
        r_pre = x_r + h_r  # (batch, hidden_size)
        # @arch grucell.forward.r_pre-x_r-h_r:end
        # @arch grucell.forward.r-torch-sigmoid-r_pre:start
        r = torch.sigmoid(r_pre)  # (batch, hidden_size)
        # @arch grucell.forward.r-torch-sigmoid-r_pre:end

        # Compute candidate state from reset hidden state: (batch, hidden_size).
        # @arch grucell.forward.reset_h-r-h:start
        reset_h = r * h  # (batch, hidden_size)
        # @arch grucell.forward.reset_h-r-h:end
        # @arch grucell.forward.x_n-self-x_n-x:start
        x_n = self.x_n(x)  # (batch, input_size) -> (batch, hidden_size)
        # @arch grucell.forward.x_n-self-x_n-x:end
        # @arch grucell.forward.h_n-self-h_n-reset_h:start
        h_n = self.h_n(reset_h)  # (batch, hidden_size)
        # @arch grucell.forward.h_n-self-h_n-reset_h:end
        # @arch grucell.forward.n_pre-x_n-h_n:start
        n_pre = x_n + h_n  # (batch, hidden_size)
        # @arch grucell.forward.n_pre-x_n-h_n:end
        # @arch grucell.forward.n-torch-tanh-n_pre:start
        n = torch.tanh(n_pre)  # (batch, hidden_size)
        # @arch grucell.forward.n-torch-tanh-n_pre:end

        # Blend previous and candidate states: (batch, hidden_size).
        # @arch grucell.forward.keep_h-z-h:start
        keep_h = z * h  # (batch, hidden_size)
        # @arch grucell.forward.keep_h-z-h:end
        # @arch grucell.forward.candidate_h-n-z-n:start
        candidate_h = (1 - z) * n  # (batch, hidden_size)
        # @arch grucell.forward.candidate_h-n-z-n:end
        # @arch grucell.forward.h_next-candidate_h-keep_h:start
        h_next = candidate_h + keep_h  # (batch, hidden_size)
        # @arch grucell.forward.h_next-candidate_h-keep_h:end
        # @arch grucell.forward.return-h_next:start
        return h_next
        # @arch grucell.forward.return-h_next:end


# %% [notebook-only]
# Create and run one GRU cell step: (2, 32), (2, 64) -> (2, 64).
example_cell = GRUCell(input_size=32, hidden_size=64)
example_cell_input = torch.randn(2, 32)  # -> (2, 32)
example_previous_state = torch.zeros(2, 64)  # -> (2, 64)
example_next_state = example_cell(example_cell_input, example_previous_state)  # (2, 32), (2, 64) -> (2, 64)
print("example_next_state shape:", example_next_state.shape)


# %%
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
        # @arch grusequence.self-readout-nn-linear-hidden_size-output_size:start
        self.readout = nn.Linear(hidden_size, output_size)
        # @arch grusequence.self-readout-nn-linear-hidden_size-output_size:end

    def forward(self, x):
        # Build the initial recurrent state: (batch, hidden_size).
        # @arch grusequence.forward.batch_size-x-size-n:start
        batch_size = x.size(0)  # (batch, steps, input_size) -> scalar
        # @arch grusequence.forward.batch_size-x-size-n:end
        # @arch grusequence.forward.hidden_shape-batch_size-self-hidden_size:start
        hidden_shape = (batch_size, self.hidden_size)  # -> (batch, hidden_size)
        # @arch grusequence.forward.hidden_shape-batch_size-self-hidden_size:end
        # @arch grusequence.forward.h-torch-zeros-hidden_shape-device-x-device:start
        h = torch.zeros(hidden_shape, device=x.device)  # -> (batch, hidden_size)
        # @arch grusequence.forward.h-torch-zeros-hidden_shape-device-x-device:end

        # Run the shared GRU cell over time: (batch, steps, input_size) -> list of (batch, hidden_size).
        # @arch grusequence.forward.states:start
        states = []
        # @arch grusequence.forward.states:end
        # @arch grusequence.forward.step_count-x-size-n:start
        step_count = x.size(1)  # (batch, steps, input_size) -> scalar
        # @arch grusequence.forward.step_count-x-size-n:end
        # @arch grusequence.forward.for-t-in-range-step_count:start
        for t in range(step_count):
        # @arch grusequence.forward.for-t-in-range-step_count:end
            # @arch grusequence.forward.current_input-x-t:start
            current_input = x[:, t]  # (batch, steps, input_size) -> (batch, input_size)
            # @arch grusequence.forward.current_input-x-t:end
            # @arch grusequence.forward.h-self-cell-current_input-h:start
            h = self.cell(current_input, h)  # (batch, input_size), (batch, hidden_size) -> (batch, hidden_size)
            # @arch grusequence.forward.h-self-cell-current_input-h:end
            # @arch grusequence.forward.states-append-h:start
            states.append(h)
            # @arch grusequence.forward.states-append-h:end

        # Project the final hidden state and pack the full state trace.
        # @arch grusequence.forward.logits-self-readout-h:start
        logits = self.readout(h)  # (batch, hidden_size) -> (batch, output_size)
        # @arch grusequence.forward.logits-self-readout-h:end
        # @arch grusequence.forward.state_trace-torch-stack-states-dim-n:start
        state_trace = torch.stack(states, dim=1)  # list of (batch, hidden_size) -> (batch, steps, hidden_size)
        # @arch grusequence.forward.state_trace-torch-stack-states-dim-n:end
        # @arch grusequence.forward.outputs-logits-state_trace:start
        outputs = (logits, state_trace)
        # @arch grusequence.forward.outputs-logits-state_trace:end
        # @arch grusequence.forward.return-outputs:start
        return outputs
        # @arch grusequence.forward.return-outputs:end


# %% [notebook-only]
# Create and run a sample sequence: (2, 8, 32) -> logits and states.
example_model = GRUSequence(input_size=32, hidden_size=64, output_size=10)
example_sequence = torch.randn(2, 8, 32)  # -> (2, 8, 32)
example_outputs = example_model(example_sequence)
example_logits = example_outputs[0]  # (2, 10)
example_states = example_outputs[1]  # (2, 8, 64)
print("example_logits shape:", example_logits.shape, "example_states shape:", example_states.shape)


# %%
# Train the same model on two synthetic sequences with opposite labels.
model = GRUSequence(input_size=32, hidden_size=64, output_size=10)
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
