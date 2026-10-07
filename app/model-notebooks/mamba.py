# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class MambaBlock(nn.Module):
    def __init__(self, model_dim=32, state_dim=16, expansion=2, kernel_size=4):
        super().__init__()
        self.inner_dim = expansion * model_dim
        self.state_dim = state_dim
        self.dt_rank = max(1, (model_dim + 15) // 16)
        self.norm_weight = nn.Parameter(torch.ones(model_dim))
        self.in_proj = nn.Linear(model_dim, 2 * self.inner_dim, bias=False)
        self.conv = nn.Conv1d(
            self.inner_dim, self.inner_dim, kernel_size,
            padding=kernel_size - 1, groups=self.inner_dim
        )
        self.select_proj = nn.Linear(self.inner_dim, self.dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_dim)
        rates = torch.arange(1, state_dim + 1, dtype=torch.float32)
        log_rates = torch.log(rates)
        self.A_log = nn.Parameter(log_rates.repeat(self.inner_dim, 1))
        self.D = nn.Parameter(torch.ones(self.inner_dim))
        self.out_proj = nn.Linear(self.inner_dim, model_dim, bias=False)
        time_steps = torch.logspace(-3, -1, self.inner_dim)
        inverse_softplus = time_steps + torch.log(-torch.expm1(-time_steps))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus)

    def forward(self, x):
        # Normalize each token independently: (batch, steps, model_dim).
        # @arch mamba.block.norm:start
        mean_square = x.square().mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + 1e-5)
        normalized = x * inverse_rms
        normalized = normalized * self.norm_weight
        # @arch mamba.block.norm:end

        # Expand features into a state branch and a gate branch.
        # @arch mamba.block.project:start
        projected = self.in_proj(normalized)  # (batch, steps, model_dim) -> (batch, steps, 2 * inner_dim)
        signal, gate = projected.chunk(2, dim=-1)  # -> two (batch, steps, inner_dim) tensors
        # @arch mamba.block.project:end

        # Apply a depthwise convolution using only current and earlier tokens.
        # @arch mamba.block.conv:start
        step_count = signal.shape[1]  # (batch, steps, inner_dim) -> scalar
        channels_first = signal.transpose(1, 2)  # (batch, steps, inner_dim) -> (batch, inner_dim, steps)
        convolved = self.conv(channels_first)
        causal = convolved[:, :, :step_count]  # -> (batch, inner_dim, steps)
        signal = causal.transpose(1, 2)  # (batch, inner_dim, steps) -> (batch, steps, inner_dim)
        signal = F.silu(signal)  # (batch, steps, inner_dim)
        # @arch mamba.block.conv:end

        # Predict token-dependent time steps and state input/output vectors.
        # @arch mamba.block.select:start
        selection = self.select_proj(signal)  # -> (batch, steps, dt_rank + 2 * state_dim)
        delta_low_rank, input_vector, output_vector = torch.split(
            selection, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        # @arch mamba.block.select:end
        # @arch mamba.block.delta:start
        delta_pre = self.dt_proj(delta_low_rank)  # (batch, steps, dt_rank) -> (batch, steps, inner_dim)
        delta = F.softplus(delta_pre)  # (batch, steps, inner_dim)
        # @arch mamba.block.delta:end

        # Discretize the stable diagonal transition for each token.
        # @arch mamba.block.discretize:start
        transition = -torch.exp(self.A_log)  # (inner_dim, state_dim)
        delta_expanded = delta.unsqueeze(-1)  # -> (batch, steps, inner_dim, 1)
        discrete_transition = torch.exp(delta_expanded * transition)  # -> (batch, steps, inner_dim, state_dim)
        input_expanded = input_vector.unsqueeze(2)  # -> (batch, steps, 1, state_dim)
        discrete_input = delta_expanded * input_expanded  # -> (batch, steps, inner_dim, state_dim)
        # @arch mamba.block.discretize:end

        # Unroll the selective recurrence; this readable reference scan uses no fused GPU kernel.
        # @arch mamba.block.scan:start
        state = signal.new_zeros(signal.shape[0], self.inner_dim, self.state_dim)
        state_outputs = []
        for step in range(step_count):
            current_input = signal[:, step, :].unsqueeze(-1)  # -> (batch, inner_dim, 1)
            retained_state = discrete_transition[:, step] * state  # (batch, inner_dim, state_dim)
            written_state = discrete_input[:, step] * current_input  # (batch, inner_dim, state_dim)
            state = retained_state + written_state  # (batch, inner_dim, state_dim)
            current_output_vector = output_vector[:, step, :].unsqueeze(1)  # -> (batch, 1, state_dim)
            read_state = state * current_output_vector  # (batch, inner_dim, state_dim)
            state_output = read_state.sum(dim=-1)  # -> (batch, inner_dim)
            state_outputs.append(state_output)
        outputs = torch.stack(state_outputs, dim=1)  # -> (batch, steps, inner_dim)
        # @arch mamba.block.scan:end

        # Add the learned direct path and modulate it with the SiLU gate.
        # @arch mamba.block.skip:start
        direct = signal * self.D  # (batch, steps, inner_dim)
        outputs = outputs + direct  # (batch, steps, inner_dim)
        # @arch mamba.block.skip:end
        # @arch mamba.block.gate:start
        activated_gate = F.silu(gate)  # (batch, steps, inner_dim)
        gated_outputs = outputs * activated_gate  # (batch, steps, inner_dim)
        # @arch mamba.block.gate:end

        # Project back to the model width and preserve the residual stream.
        # @arch mamba.block.output:start
        mixed = self.out_proj(gated_outputs)  # -> (batch, steps, model_dim)
        # @arch mamba.block.output:end
        # @arch mamba.block.residual:start
        outputs = x + mixed  # (batch, steps, model_dim)
        # @arch mamba.block.residual:end
        return outputs


# %% [notebook-only]
example_block = MambaBlock(model_dim=16, state_dim=8)
example_inputs = torch.randn(2, 8, 16)  # -> (2, 8, 16)
example_outputs = example_block(example_inputs)  # (2, 8, 16)
print("Mamba block:", example_inputs.shape, "->", example_outputs.shape)


# %%
class MambaLM(nn.Module):
    def __init__(self, vocab_size=32, model_dim=32, state_dim=16, layer_count=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([
            MambaBlock(model_dim=model_dim, state_dim=state_dim) for _ in range(layer_count)
        ])
        self.final_norm_weight = nn.Parameter(torch.ones(model_dim))

    def forward(self, tokens):
        # Consume token ids through the learned embedding table.
        # @arch mamba.embedding:start
        hidden = self.embedding(tokens)  # (batch, steps) -> (batch, steps, model_dim)
        # @arch mamba.embedding:end
        for block in self.blocks:
            hidden = block(hidden)  # (batch, steps, model_dim)

        # Normalize the final residual stream before vocabulary prediction.
        # @arch mamba.final_norm:start
        mean_square = hidden.square().mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + 1e-5)
        normalized = hidden * inverse_rms
        normalized = normalized * self.final_norm_weight
        # @arch mamba.final_norm:end

        # Reuse the embedding weights as the vocabulary readout.
        # @arch mamba.logits:start
        logits = F.linear(normalized, self.embedding.weight)  # -> (batch, steps, vocab_size)
        # @arch mamba.logits:end
        return logits


# %% [notebook-only]
example_model = MambaLM(vocab_size=16, model_dim=16, state_dim=8)
example_tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
example_logits = example_model(example_tokens)  # -> (2, 4, 16)
print("Mamba language model:", example_tokens.shape, "->", example_logits.shape)


# %%
# Fit next-token predictions on a tiny synthetic batch.
model = MambaLM(vocab_size=16, model_dim=16, state_dim=8)
train_tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
train_targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])  # -> (2, 4)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for step in range(3):
    optimizer.zero_grad()
    logits = model(train_tokens)  # -> (2, 4, 16)
    flat_logits = logits.reshape(-1, 16)  # -> (8, 16)
    flat_targets = train_targets.reshape(-1)  # -> (8)
    loss = F.cross_entropy(flat_logits, flat_targets)  # -> scalar
    loss.backward()
    optimizer.step()
final_loss = loss.item()  # scalar
