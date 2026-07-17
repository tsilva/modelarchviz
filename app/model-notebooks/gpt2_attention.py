# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
# @arch class-causalselfattention-nn-module:start
class CausalSelfAttention(nn.Module):
# @arch class-causalselfattention-nn-module:end
    # @arch causalselfattention.def-__init__:start
    def __init__(
    # @arch causalselfattention.def-__init__:end
        # @arch causalselfattention.self:start
        self,
        # @arch causalselfattention.self:end
        # @arch causalselfattention.n_embd-n:start
        n_embd=768,  # Embedding width.
        # @arch causalselfattention.n_embd-n:end
        # @arch causalselfattention.n_head-n:start
        n_head=12  # Number of attention heads.
        # @arch causalselfattention.n_head-n:end
    # @arch causalselfattention.code:start
    ):
    # @arch causalselfattention.code:end
        # @arch causalselfattention.super-__init__:start
        super().__init__()
        # @arch causalselfattention.super-__init__:end

        # Register packed QKV projection and output projection.
        # @arch causalselfattention.self-n_head-n_head:start
        self.n_head = n_head
        # @arch causalselfattention.self-n_head-n_head:end
        # @arch causalselfattention.self-c_attn-nn-linear-n_embd-n-n_embd:start
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # @arch causalselfattention.self-c_attn-nn-linear-n_embd-n-n_embd:end
        # @arch causalselfattention.self-c_proj-nn-linear-n_embd-n_embd:start
        self.c_proj = nn.Linear(n_embd, n_embd)
        # @arch causalselfattention.self-c_proj-nn-linear-n_embd-n_embd:end

    # @arch causalselfattention.def-forward-self-x-mask:start
    def forward(self, x, mask):
    # @arch causalselfattention.def-forward-self-x-mask:end
        # Project hidden states into query, key, and value tensors.
        # @arch causalselfattention.forward.batch_size-step_count-channel_count-x-shape:start
        batch_size, step_count, channel_count = x.shape  # (batch, steps, channels) -> scalar, scalar, scalar
        # @arch causalselfattention.forward.batch_size-step_count-channel_count-x-shape:end
        # @arch causalselfattention.forward.qkv-self-c_attn-x:start
        qkv = self.c_attn(x)  # (batch, steps, channels) -> (batch, steps, 3 * channels)
        # @arch causalselfattention.forward.qkv-self-c_attn-x:end
        # @arch causalselfattention.forward.q-k-v-qkv-split-channel_count-dim-n:start
        q, k, v = qkv.split(channel_count, dim=2)  # (batch, steps, 3 * channels) -> three (batch, steps, channels)
        # @arch causalselfattention.forward.q-k-v-qkv-split-channel_count-dim-n:end
        # @arch causalselfattention.forward.head_dim-channel_count-self-n_head:start
        head_dim = channel_count // self.n_head  # scalar
        # @arch causalselfattention.forward.head_dim-channel_count-self-n_head:end

        # Split heads: (batch, steps, channels) -> (batch, heads, steps, head_dim).
        # @arch causalselfattention.forward.q-q-view-batch_size-step_count-self-n_head-head_dim:start
        q = q.view(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.forward.q-q-view-batch_size-step_count-self-n_head-head_dim:end
        # @arch causalselfattention.forward.q-q-transpose-n-n:start
        q = q.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.forward.q-q-transpose-n-n:end
        # @arch causalselfattention.forward.k-k-view-batch_size-step_count-self-n_head-head_dim:start
        k = k.view(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.forward.k-k-view-batch_size-step_count-self-n_head-head_dim:end
        # @arch causalselfattention.forward.k-k-transpose-n-n:start
        k = k.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.forward.k-k-transpose-n-n:end
        # @arch causalselfattention.forward.v-v-view-batch_size-step_count-self-n_head-head_dim:start
        v = v.view(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.forward.v-v-view-batch_size-step_count-self-n_head-head_dim:end
        # @arch causalselfattention.forward.v-v-transpose-n-n:start
        v = v.transpose(1, 2)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.forward.v-v-transpose-n-n:end

        # Compute masked causal attention weights.
        # @arch causalselfattention.forward.key_transpose-k-transpose-n-n:start
        key_transpose = k.transpose(-2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        # @arch causalselfattention.forward.key_transpose-k-transpose-n-n:end
        # @arch causalselfattention.forward.scores-q-key_transpose:start
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        # @arch causalselfattention.forward.scores-q-key_transpose:end
        # @arch causalselfattention.forward.scale-k-size-n-n:start
        scale = k.size(-1) ** -0.5  # (batch, heads, steps, head_dim) -> scalar
        # @arch causalselfattention.forward.scale-k-size-n-n:end
        # @arch causalselfattention.forward.att-scores-scale:start
        att = scores * scale  # (batch, heads, steps, steps)
        # @arch causalselfattention.forward.att-scores-scale:end
        # @arch causalselfattention.forward.mask_window-mask-step_count-step_count:start
        mask_window = mask[:, :, :step_count, :step_count]  # (1, 1, max_steps, max_steps) -> (1, 1, steps, steps)
        # @arch causalselfattention.forward.mask_window-mask-step_count-step_count:end
        # @arch causalselfattention.forward.att-att-masked_fill-mask_window-n-float-inf:start
        att = att.masked_fill(mask_window == 0, float('-inf'))  # (batch, heads, steps, steps)
        # @arch causalselfattention.forward.att-att-masked_fill-mask_window-n-float-inf:end
        # @arch causalselfattention.forward.weights-f-softmax-att-dim-n:start
        weights = F.softmax(att, dim=-1)  # (batch, heads, steps, steps)
        # @arch causalselfattention.forward.weights-f-softmax-att-dim-n:end

        # Merge heads back to the model width and project.
        # @arch causalselfattention.forward.y-weights-v:start
        y = weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.forward.y-weights-v:end
        # @arch causalselfattention.forward.y-y-transpose-n-n:start
        y = y.transpose(1, 2)  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.forward.y-y-transpose-n-n:end
        # @arch causalselfattention.forward.y-y-contiguous:start
        y = y.contiguous()  # (batch, steps, heads, head_dim)
        # @arch causalselfattention.forward.y-y-contiguous:end
        # @arch causalselfattention.forward.y-y-view-batch_size-step_count-channel_count:start
        y = y.view(batch_size, step_count, channel_count)  # (batch, steps, heads, head_dim) -> (batch, steps, channels)
        # @arch causalselfattention.forward.y-y-view-batch_size-step_count-channel_count:end
        # @arch causalselfattention.forward.out-self-c_proj-y:start
        out = self.c_proj(y)  # (batch, steps, channels)
        # @arch causalselfattention.forward.out-self-c_proj-y:end
        # @arch causalselfattention.forward.return-out:start
        return out  # (batch, steps, channels)
        # @arch causalselfattention.forward.return-out:end


# %% [notebook-only]
# Create and run causal self-attention: (2, 4, 24) -> (2, 4, 24).
example_attention = CausalSelfAttention(n_embd=24, n_head=4)
example_hidden_states = torch.randn(2, 4, 24)  # -> (2, 4, 24)
example_mask = torch.tril(torch.ones(4, 4)).view(1, 1, 4, 4)  # -> (1, 1, 4, 4)
example_attended = example_attention(example_hidden_states, example_mask)  # (2, 4, 24), (1, 1, 4, 4) -> (2, 4, 24)
print("attended shape:", example_attended.shape)

# %%
# @arch class-block-nn-module:start
class Block(nn.Module):
# @arch class-block-nn-module:end
    # @arch block.def-__init__-self:start
    def __init__(self):
    # @arch block.def-__init__-self:end
        # @arch block.__init__.super-__init__:start
        super().__init__()
        # @arch block.__init__.super-__init__:end

        # Register pre-normalized attention and MLP sublayers.
        # @arch block.__init__.self-ln_n-nn-layernorm-n:start
        self.ln_1 = nn.LayerNorm(768)
        # @arch block.__init__.self-ln_n-nn-layernorm-n:end
        # @arch block.__init__.self-attn-causalselfattention:start
        self.attn = CausalSelfAttention()
        # @arch block.__init__.self-attn-causalselfattention:end
        # @arch block.__init__.self-ln_n-nn-layernorm-n.2:start
        self.ln_2 = nn.LayerNorm(768)
        # @arch block.__init__.self-ln_n-nn-layernorm-n.2:end
        # @arch block.__init__.self-mlp-nn-sequential:start
        self.mlp = nn.Sequential(
        # @arch block.__init__.self-mlp-nn-sequential:end
            # @arch block.__init__.nn-linear-n-n:start
            nn.Linear(768, 3072),
            # @arch block.__init__.nn-linear-n-n:end
            # @arch block.__init__.nn-gelu:start
            nn.GELU(),
            # @arch block.__init__.nn-gelu:end
            # @arch block.__init__.nn-linear-n-n.2:start
            nn.Linear(3072, 768),
            # @arch block.__init__.nn-linear-n-n.2:end
        # @arch block.__init__.code.3:start
        )
        # @arch block.__init__.code.3:end

    # @arch block.def-forward-self-x-mask:start
    def forward(self, x, mask):
    # @arch block.def-forward-self-x-mask:end
        # Apply causal attention with a residual connection.
        # @arch block.forward.attn_input-self-ln_n-x:start
        attn_input = self.ln_1(x)  # (batch, steps, 768)
        # @arch block.forward.attn_input-self-ln_n-x:end
        # @arch block.forward.attn-self-attn-attn_input-mask:start
        attn = self.attn(attn_input, mask)  # (batch, steps, 768)
        # @arch block.forward.attn-self-attn-attn_input-mask:end
        # @arch block.forward.x-x-attn:start
        x = x + attn  # (batch, steps, 768)
        # @arch block.forward.x-x-attn:end

        # Apply MLP with a residual connection.
        # @arch block.forward.mlp_input-self-ln_n-x:start
        mlp_input = self.ln_2(x)  # (batch, steps, 768)
        # @arch block.forward.mlp_input-self-ln_n-x:end
        # @arch block.forward.mlp_out-self-mlp-mlp_input:start
        mlp_out = self.mlp(mlp_input)  # (batch, steps, 768)
        # @arch block.forward.mlp_out-self-mlp-mlp_input:end
        # @arch block.forward.x-x-mlp_out:start
        x = x + mlp_out  # (batch, steps, 768)
        # @arch block.forward.x-x-mlp_out:end
        # @arch block.forward.return-x:start
        return x  # (batch, steps, 768)
        # @arch block.forward.return-x:end


# %% [notebook-only]
# Create and run one GPT block: (2, 4, 768) -> (2, 4, 768).
example_block = Block()
example_hidden_states = torch.randn(2, 4, 768)  # -> (2, 4, 768)
example_mask_values = torch.ones(4, 4)  # -> (4, 4)
example_mask = torch.tril(example_mask_values)  # (4, 4)
example_mask = example_mask.view(1, 1, 4, 4)  # (4, 4) -> (1, 1, 4, 4)
example_block_output = example_block(example_hidden_states, example_mask)  # (2, 4, 768), (1, 1, 4, 4) -> (2, 4, 768)
print("block output shape:", example_block_output.shape)

# %%
class GPT2Small(nn.Module):
    def __init__(
        self,
        vocab_size,  # Number of token ids.
        n_ctx=1024,  # Maximum context length.
        n_embd=768  # Embedding width.
    ):
        super().__init__()

        # Register embeddings, transformer blocks, final norm, and language-model head.
        # @arch gptnsmall.self-wte-nn-embedding-vocab_size-n_embd:start
        self.wte = nn.Embedding(vocab_size, n_embd)
        # @arch gptnsmall.self-wte-nn-embedding-vocab_size-n_embd:end
        # @arch gptnsmall.self-wpe-nn-embedding-n_ctx-n_embd:start
        self.wpe = nn.Embedding(n_ctx, n_embd)
        # @arch gptnsmall.self-wpe-nn-embedding-n_ctx-n_embd:end
        # @arch gptnsmall.self-drop-nn-dropout-n:start
        self.drop = nn.Dropout(0.1)
        # @arch gptnsmall.self-drop-nn-dropout-n:end
        # @arch gptnsmall.self-blocks-nn-modulelist-block-for-_-in-range-n:start
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        # @arch gptnsmall.self-blocks-nn-modulelist-block-for-_-in-range-n:end
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, mask):
        # Combine token and position embeddings: (batch, steps) -> (batch, steps, n_embd).
        batch_size, step_count = input_ids.shape  # (batch, steps) -> scalar, scalar
        # @arch gptnsmall.forward.positions-torch-arange-step_count-device-input_ids-device:start
        positions = torch.arange(step_count, device=input_ids.device)  # -> (steps)
        # @arch gptnsmall.forward.positions-torch-arange-step_count-device-input_ids-device:end
        # @arch gptnsmall.forward.token_embeddings-self-wte-input_ids:start
        token_embeddings = self.wte(input_ids)  # (batch, steps) -> (batch, steps, n_embd)
        # @arch gptnsmall.forward.token_embeddings-self-wte-input_ids:end
        # @arch gptnsmall.forward.position_embeddings-self-wpe-positions:start
        position_embeddings = self.wpe(positions)  # (steps) -> (steps, n_embd)
        # @arch gptnsmall.forward.position_embeddings-self-wpe-positions:end
        # @arch gptnsmall.forward.position_embeddings-position_embeddings-none:start
        position_embeddings = position_embeddings[None, :, :]  # (steps, n_embd) -> (1, steps, n_embd)
        # @arch gptnsmall.forward.position_embeddings-position_embeddings-none:end
        x = token_embeddings + position_embeddings  # (batch, steps, n_embd)
        # @arch gptnsmall.forward.x-self-drop-x:start
        x = self.drop(x)  # (batch, steps, n_embd)
        # @arch gptnsmall.forward.x-self-drop-x:end

        # Run the transformer block stack while preserving sequence shape.
        # @arch gptnsmall.forward.for-block-in-self-blocks:start
        for block in self.blocks:
        # @arch gptnsmall.forward.for-block-in-self-blocks:end
            # @arch gptnsmall.forward.x-block-x-mask:start
            x = block(x, mask)  # (batch, steps, n_embd)
            # @arch gptnsmall.forward.x-block-x-mask:end

        # Normalize final states and project to vocabulary logits.
        x = self.ln_f(x)  # (batch, steps, n_embd)
        logits = self.lm_head(x)  # (batch, steps, n_embd) -> (batch, steps, vocab_size)
        return logits  # (batch, steps, vocab_size)


# %% [notebook-only]
# Create and run a sample token batch.
example_model = GPT2Small(vocab_size=50257)
example_test_input = torch.randint(0, 50257, (2, 16))  # -> (2, 16)

# Build a causal attention mask: (1, 1, 16, 16).
mask_values = torch.ones(16, 16)  # -> (16, 16)
example_mask = torch.tril(mask_values)  # (16, 16)
example_mask = example_mask.view(1, 1, 16, 16)  # (16, 16) -> (1, 1, 16, 16)
example_logits = example_model(example_test_input, example_mask)  # (2, 16), (1, 1, 16, 16) -> (2, 16, 50257)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny next-token prediction batch.
model = GPT2Small(vocab_size=20)
input_ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])  # -> (2, 4)
train_targets = torch.tensor([[2, 3, 4, 5], [3, 2, 1, 0]])  # -> (2, 4)
mask_values = torch.ones(4, 4)  # -> (4, 4)
mask = torch.tril(mask_values)  # (4, 4)
mask = mask.view(1, 1, 4, 4)  # (4, 4) -> (1, 1, 4, 4)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    optimizer.zero_grad()
    logits = model(input_ids, mask)  # (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
    flat_logits = logits.reshape(-1, logits.size(-1))  # (2, 4, 20) -> (8, 20)
    flat_targets = train_targets.reshape(-1)  # (2, 4) -> (8)
    loss = criterion(flat_logits, flat_targets)  # (8, 20), (8) -> scalar
    loss.backward()
    optimizer.step()

# Keep the final scalar loss for inspection.
final_loss = loss.item()  # scalar
