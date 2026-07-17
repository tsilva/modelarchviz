# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
# @arch class-causalselfattention-nn-module:start
class CausalSelfAttention(nn.Module):
# @arch class-causalselfattention-nn-module:end
    # @arch causalselfattention.n_embd-int-n:start
    n_embd: int = 768
    # @arch causalselfattention.n_embd-int-n:end
    # @arch causalselfattention.n_head-int-n:start
    n_head: int = 12
    # @arch causalselfattention.n_head-int-n:end

    # @arch causalselfattention.nn-compact:start
    @nn.compact
    # @arch causalselfattention.nn-compact:end
    # @arch causalselfattention.def-__call__-self-x-mask:start
    def __call__(self, x, mask):
    # @arch causalselfattention.def-__call__-self-x-mask:end
        # Project hidden states into query, key, and value tensors.
        # @arch causalselfattention.__call__.batch_size-step_count-channel_count-x-shape:start
        batch_size, step_count, channel_count = x.shape  # (batch, steps, channels) -> scalar, scalar, scalar
        # @arch causalselfattention.__call__.batch_size-step_count-channel_count-x-shape:end
        # @arch causalselfattention.__call__.qkv_dim-n-channel_count:start
        qkv_dim = 3 * channel_count  # scalar
        # @arch causalselfattention.__call__.qkv_dim-n-channel_count:end
        # @arch causalselfattention.__call__.qkv-nn-dense-qkv_dim-name-c_attn-x:start
        qkv = nn.Dense(qkv_dim, name='c_attn')(x)  # (batch, steps, channels) -> (batch, steps, 3 * channels)
        # @arch causalselfattention.__call__.qkv-nn-dense-qkv_dim-name-c_attn-x:end
        # @arch causalselfattention.__call__.q-k-v-jnp-split-qkv-n-axis-n:start
        q, k, v = jnp.split(qkv, 3, axis=2)  # (batch, steps, 3 * channels) -> three (batch, steps, channels)
        # @arch causalselfattention.__call__.q-k-v-jnp-split-qkv-n-axis-n:end
        # @arch causalselfattention.__call__.head_dim-channel_count-self-n_head:start
        head_dim = channel_count // self.n_head  # scalar
        # @arch causalselfattention.__call__.head_dim-channel_count-self-n_head:end

        # Split heads: (batch, steps, channels) -> (batch, heads, steps, head_dim).
        # @arch causalselfattention.__call__.q-q-reshape-batch_size-step_count-self-n_head-head_dim:start
        q = q.reshape(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.__call__.q-q-reshape-batch_size-step_count-self-n_head-head_dim:end
        # @arch causalselfattention.__call__.q-q-transpose-n-n-n-n:start
        q = q.transpose(0, 2, 1, 3)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.__call__.q-q-transpose-n-n-n-n:end
        # @arch causalselfattention.__call__.k-k-reshape-batch_size-step_count-self-n_head-head_dim:start
        k = k.reshape(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.__call__.k-k-reshape-batch_size-step_count-self-n_head-head_dim:end
        # @arch causalselfattention.__call__.k-k-transpose-n-n-n-n:start
        k = k.transpose(0, 2, 1, 3)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.__call__.k-k-transpose-n-n-n-n:end
        # @arch causalselfattention.__call__.v-v-reshape-batch_size-step_count-self-n_head-head_dim:start
        v = v.reshape(batch_size, step_count, self.n_head, head_dim)  # (batch, steps, channels) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.__call__.v-v-reshape-batch_size-step_count-self-n_head-head_dim:end
        # @arch causalselfattention.__call__.v-v-transpose-n-n-n-n:start
        v = v.transpose(0, 2, 1, 3)  # (batch, steps, heads, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.__call__.v-v-transpose-n-n-n-n:end

        # Compute masked causal attention weights.
        # @arch causalselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n:start
        key_transpose = jnp.swapaxes(k, -2, -1)  # (batch, heads, steps, head_dim) -> (batch, heads, head_dim, steps)
        # @arch causalselfattention.__call__.key_transpose-jnp-swapaxes-k-n-n:end
        # @arch causalselfattention.__call__.scores-q-key_transpose:start
        scores = q @ key_transpose  # (batch, heads, steps, head_dim), (batch, heads, head_dim, steps) -> (batch, heads, steps, steps)
        # @arch causalselfattention.__call__.scores-q-key_transpose:end
        # @arch causalselfattention.__call__.scale-k-shape-n-n:start
        scale = k.shape[-1] ** -0.5  # (batch, heads, steps, head_dim) -> scalar
        # @arch causalselfattention.__call__.scale-k-shape-n-n:end
        # @arch causalselfattention.__call__.att-scores-scale:start
        att = scores * scale  # (batch, heads, steps, steps)
        # @arch causalselfattention.__call__.att-scores-scale:end
        # @arch causalselfattention.__call__.mask_window-mask-step_count-step_count:start
        mask_window = mask[:, :, :step_count, :step_count]  # (1, 1, max_steps, max_steps) -> (1, 1, steps, steps)
        # @arch causalselfattention.__call__.mask_window-mask-step_count-step_count:end
        # @arch causalselfattention.__call__.att-jnp-where-mask_window-n-jnp-inf-att:start
        att = jnp.where(mask_window == 0, -jnp.inf, att)  # (batch, heads, steps, steps)
        # @arch causalselfattention.__call__.att-jnp-where-mask_window-n-jnp-inf-att:end
        # @arch causalselfattention.__call__.weights-nn-softmax-att-axis-n:start
        weights = nn.softmax(att, axis=-1)  # (batch, heads, steps, steps)
        # @arch causalselfattention.__call__.weights-nn-softmax-att-axis-n:end

        # Merge heads back to the model width and project.
        # @arch causalselfattention.__call__.y-weights-v:start
        y = weights @ v  # (batch, heads, steps, steps), (batch, heads, steps, head_dim) -> (batch, heads, steps, head_dim)
        # @arch causalselfattention.__call__.y-weights-v:end
        # @arch causalselfattention.__call__.y-y-transpose-n-n-n-n:start
        y = y.transpose(0, 2, 1, 3)  # (batch, heads, steps, head_dim) -> (batch, steps, heads, head_dim)
        # @arch causalselfattention.__call__.y-y-transpose-n-n-n-n:end
        # @arch causalselfattention.__call__.y-y-reshape-batch_size-step_count-channel_count:start
        y = y.reshape(batch_size, step_count, channel_count)  # (batch, steps, heads, head_dim) -> (batch, steps, channels)
        # @arch causalselfattention.__call__.y-y-reshape-batch_size-step_count-channel_count:end
        # @arch causalselfattention.__call__.out-nn-dense-channel_count-name-c_proj-y:start
        out = nn.Dense(channel_count, name='c_proj')(y)  # (batch, steps, channels)
        # @arch causalselfattention.__call__.out-nn-dense-channel_count-name-c_proj-y:end
        # @arch causalselfattention.__call__.return-out:start
        return out  # (batch, steps, channels)
        # @arch causalselfattention.__call__.return-out:end


# %% [notebook-only]
# Create and run causal self-attention: (2, 4, 24) -> (2, 4, 24).
example_attention = CausalSelfAttention(n_embd=24, n_head=4)
example_hidden_states = jnp.ones((2, 4, 24))  # -> (2, 4, 24)
example_mask = jnp.tril(jnp.ones((4, 4))).reshape(1, 1, 4, 4)  # -> (1, 1, 4, 4)
example_params = example_attention.init(jax.random.PRNGKey(0), example_hidden_states, example_mask)
example_attended = example_attention.apply(example_params, example_hidden_states, example_mask)  # (2, 4, 24), (1, 1, 4, 4) -> (2, 4, 24)
print("attended shape:", example_attended.shape)

# %%
# @arch class-mlp-nn-module:start
class MLP(nn.Module):
# @arch class-mlp-nn-module:end
    # @arch mlp.n_embd-int-n:start
    n_embd: int = 768
    # @arch mlp.n_embd-int-n:end
    # @arch mlp.hidden_dim-int-n:start
    hidden_dim: int = 3072
    # @arch mlp.hidden_dim-int-n:end

    # @arch mlp.nn-compact:start
    @nn.compact
    # @arch mlp.nn-compact:end
    # @arch mlp.def-__call__-self-x:start
    def __call__(self, x):
    # @arch mlp.def-__call__-self-x:end
        # Expand and contract hidden states through the GPT feed-forward block.
        # @arch mlp.__call__.hidden-nn-dense-self-hidden_dim-name-c_fc-x:start
        hidden = nn.Dense(self.hidden_dim, name='c_fc')(x)  # (batch, steps, n_embd) -> (batch, steps, hidden_dim)
        # @arch mlp.__call__.hidden-nn-dense-self-hidden_dim-name-c_fc-x:end
        # @arch mlp.__call__.hidden-nn-gelu-hidden:start
        hidden = nn.gelu(hidden)  # (batch, steps, hidden_dim)
        # @arch mlp.__call__.hidden-nn-gelu-hidden:end
        # @arch mlp.__call__.out-nn-dense-self-n_embd-name-c_proj-hidden:start
        out = nn.Dense(self.n_embd, name='c_proj')(hidden)  # (batch, steps, hidden_dim) -> (batch, steps, n_embd)
        # @arch mlp.__call__.out-nn-dense-self-n_embd-name-c_proj-hidden:end
        # @arch mlp.__call__.return-out:start
        return out  # (batch, steps, n_embd)
        # @arch mlp.__call__.return-out:end


# %% [notebook-only]
# Create and run the GPT feed-forward block: (2, 4, 24) -> (2, 4, 24).
example_mlp = MLP(n_embd=24, hidden_dim=48)
example_hidden_states = jnp.ones((2, 4, 24))  # -> (2, 4, 24)
example_params = example_mlp.init(jax.random.PRNGKey(1), example_hidden_states)
example_mlp_output = example_mlp.apply(example_params, example_hidden_states)  # (2, 4, 24) -> (2, 4, 24)
print("mlp_output shape:", example_mlp_output.shape)

# %%
# @arch class-block-nn-module:start
class Block(nn.Module):
# @arch class-block-nn-module:end
    # @arch block.nn-compact:start
    @nn.compact
    # @arch block.nn-compact:end
    # @arch block.def-__call__-self-x-mask:start
    def __call__(self, x, mask):
    # @arch block.def-__call__-self-x-mask:end
        # Apply causal attention with a residual connection.
        # @arch block.__call__.attn_input-nn-layernorm-name-ln_n-x:start
        attn_input = nn.LayerNorm(name='ln_1')(x)  # (batch, steps, 768)
        # @arch block.__call__.attn_input-nn-layernorm-name-ln_n-x:end
        # @arch block.__call__.attn-causalselfattention-attn_input-mask:start
        attn = CausalSelfAttention()(attn_input, mask)  # (batch, steps, 768)
        # @arch block.__call__.attn-causalselfattention-attn_input-mask:end
        # @arch block.__call__.x-x-attn:start
        x = x + attn  # (batch, steps, 768)
        # @arch block.__call__.x-x-attn:end

        # Apply MLP with a residual connection.
        # @arch block.__call__.mlp_input-nn-layernorm-name-ln_n-x:start
        mlp_input = nn.LayerNorm(name='ln_2')(x)  # (batch, steps, 768)
        # @arch block.__call__.mlp_input-nn-layernorm-name-ln_n-x:end
        # @arch block.__call__.mlp_out-mlp-mlp_input:start
        mlp_out = MLP()(mlp_input)  # (batch, steps, 768)
        # @arch block.__call__.mlp_out-mlp-mlp_input:end
        # @arch block.__call__.x-x-mlp_out:start
        x = x + mlp_out  # (batch, steps, 768)
        # @arch block.__call__.x-x-mlp_out:end
        # @arch block.__call__.return-x:start
        return x  # (batch, steps, 768)
        # @arch block.__call__.return-x:end


# %% [notebook-only]
# Create and run one GPT block: (2, 4, 768) -> (2, 4, 768).
example_block = Block()
example_hidden_states = jnp.ones((2, 4, 768))  # -> (2, 4, 768)
example_mask_values = jnp.ones((4, 4))  # -> (4, 4)
example_mask = jnp.tril(example_mask_values)  # (4, 4)
example_mask = example_mask.reshape(1, 1, 4, 4)  # (4, 4) -> (1, 1, 4, 4)
example_params = example_block.init(jax.random.PRNGKey(2), example_hidden_states, example_mask)
example_block_output = example_block.apply(example_params, example_hidden_states, example_mask)  # (2, 4, 768), (1, 1, 4, 4) -> (2, 4, 768)
print("block output shape:", example_block_output.shape)

# %%
class GPT2Small(nn.Module):
    vocab_size: int
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12

    @nn.compact
    def __call__(self, input_ids, mask):
        # Combine token and position embeddings: (batch, steps) -> (batch, steps, n_embd).
        batch_size, step_count = input_ids.shape  # (batch, steps) -> scalar, scalar
        # @arch gptnsmall.__call__.positions-jnp-arange-step_count:start
        positions = jnp.arange(step_count)  # -> (steps)
        # @arch gptnsmall.__call__.positions-jnp-arange-step_count:end
        # @arch gptnsmall.__call__.token_embeddings-nn-embed-self-vocab_size-self-n_embd-name-wte-input_ids:start
        token_embeddings = nn.Embed(self.vocab_size, self.n_embd, name='wte')(input_ids)  # (batch, steps) -> (batch, steps, n_embd)
        # @arch gptnsmall.__call__.token_embeddings-nn-embed-self-vocab_size-self-n_embd-name-wte-input_ids:end
        # @arch gptnsmall.__call__.position_embeddings-nn-embed-self-n_ctx-self-n_embd-name-wpe-positions:start
        position_embeddings = nn.Embed(self.n_ctx, self.n_embd, name='wpe')(positions)  # (steps) -> (steps, n_embd)
        # @arch gptnsmall.__call__.position_embeddings-nn-embed-self-n_ctx-self-n_embd-name-wpe-positions:end
        # @arch gptnsmall.__call__.position_embeddings-position_embeddings-none:start
        position_embeddings = position_embeddings[None, :, :]  # (steps, n_embd) -> (1, steps, n_embd)
        # @arch gptnsmall.__call__.position_embeddings-position_embeddings-none:end
        x = token_embeddings + position_embeddings  # (batch, steps, n_embd)
        # @arch gptnsmall.__call__.x-nn-dropout-n-deterministic-true-name-drop-x:start
        x = nn.Dropout(0.1, deterministic=True, name='drop')(x)  # (batch, steps, n_embd)
        # @arch gptnsmall.__call__.x-nn-dropout-n-deterministic-true-name-drop-x:end

        # Run the transformer block stack while preserving sequence shape.
        # @arch gptnsmall.__call__.for-_-in-range-self-n_layer:start
        for _ in range(self.n_layer):
        # @arch gptnsmall.__call__.for-_-in-range-self-n_layer:end
            # @arch gptnsmall.__call__.x-block-x-mask:start
            x = Block()(x, mask)  # (batch, steps, n_embd)
            # @arch gptnsmall.__call__.x-block-x-mask:end

        # Normalize final states and project to vocabulary logits.
        x = nn.LayerNorm(name='ln_f')(x)  # (batch, steps, n_embd)
        logits = nn.Dense(self.vocab_size, name='lm_head')(x)  # (batch, steps, n_embd) -> (batch, steps, vocab_size)
        return logits  # (batch, steps, vocab_size)


# %% [notebook-only]
# Create and run a sample token batch.
example_model = GPT2Small(vocab_size=50257)
example_test_input = jnp.ones((2, 16), dtype=jnp.int32)  # -> (2, 16)

# Build a causal attention mask: (1, 1, 16, 16).
mask_values = jnp.ones((16, 16))  # -> (16, 16)
example_mask = jnp.tril(mask_values)  # (16, 16)
example_mask = example_mask.reshape(1, 1, 16, 16)  # (16, 16) -> (1, 1, 16, 16)
example_params = example_model.init(jax.random.PRNGKey(0), example_test_input, example_mask)
example_logits = example_model.apply(example_params, example_test_input, example_mask)  # (2, 16), (1, 1, 16, 16) -> (2, 16, 50257)
print("logits shape:", example_logits.shape)

# %%
# Train on a tiny next-token prediction batch.
model = GPT2Small(vocab_size=20, n_layer=1)
input_ids = jnp.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=jnp.int32)  # -> (2, 4)
train_targets = jnp.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=jnp.int32)  # -> (2, 4)
mask_values = jnp.ones((4, 4))  # -> (4, 4)
mask = jnp.tril(mask_values)  # (4, 4)
mask = mask.reshape(1, 1, 4, 4)  # (4, 4) -> (1, 1, 4, 4)
params = model.init(jax.random.PRNGKey(1), input_ids, mask)


def train_step(params, inputs, mask, targets, learning_rate=0.01):
    def loss_fn(current_params):
        logits = model.apply(current_params, inputs, mask)  # (2, 4), (1, 1, 4, 4) -> (2, 4, 20)
        one_hot_targets = jax.nn.one_hot(targets, logits.shape[-1])  # (2, 4) -> (2, 4, 20)
        log_probs = jax.nn.log_softmax(logits, axis=-1)  # (2, 4, 20)
        loss = -jnp.mean(jnp.sum(one_hot_targets * log_probs, axis=-1))  # (2, 4, 20), (2, 4, 20) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model for a few steps on the tiny dataset.
for step in range(3):
    params, loss = train_step(params, input_ids, mask, train_targets)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
