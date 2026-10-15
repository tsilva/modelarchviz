# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# %% [markdown]
# # EmbeddingGemma 2
# Architecture implementation with random weights; this does not load the trained checkpoint.
# Based on Google's released config and Hugging Face's Apache-2.0 implementation.
# Inputs are processor-produced token IDs, flattened RGB patches in [0, 1], patch (x, y)
# coordinates, and 128-bin audio features. Tokenization, media decoding, and feature
# extraction are outside this notebook. Video frames use the vision branch.
# Defaults describe the released model; smoke tests shrink widths and depths only.
# Use float32 or bfloat16, not float16. See docs/embeddinggemma2.md for provenance.


# %%
TEXT_CONFIG = dict(vocab=262144, width=512, layers=24, hidden=2048, heads=4,
                   local_kv=2, local_dim=256, global_dim=512, radius=512,
                   ple_width=512, output=768, image_token=258880, audio_token=258881, video_token=258884)
VISION_CONFIG = dict(width=768, layers=16, hidden=3072, heads=12, head_dim=64,
                     patch=16, positions=10240, pool=3)
AUDIO_CONFIG = dict(width=1024, layers=12, heads=8, channels=(128, 32),
                    bins=128, kernel=5, chunk=12, left=13, right=0, output=1536)


# %% [notebook-only]
example_text_config = dict(TEXT_CONFIG, vocab=64, width=16, layers=6, hidden=32,
                           heads=4, local_kv=2, local_dim=4, global_dim=8,
                           radius=2, ple_width=8, output=12, image_token=61, audio_token=62, video_token=63)
example_vision_config = dict(VISION_CONFIG, width=16, layers=1, hidden=32,
                             heads=4, head_dim=4, patch=2, positions=16, pool=2)
example_audio_config = dict(AUDIO_CONFIG, width=16, layers=1, heads=4,
                            channels=(4, 2), bins=8, chunk=4, left=5, output=24)


# %%
class RMSNorm(nn.Module):
    def __init__(self, width, scale=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width)) if scale else None

    def forward(self, x):
        # Accumulate normalization in float32: (batch, steps, width).
        # @arch eg.norm:start
        values = x.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.pow(mean_square + 1e-6, -0.5)
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        return normalized.to(x.dtype)
        # @arch eg.norm:end


# %% [notebook-only]
example_model = RMSNorm(16)
example_outputs = example_model(torch.ones(1, 3, 16))
print('RMSNorm: (1, 3, 16) ->', tuple(example_outputs.shape))


# %%
class ClippedLinear(nn.Module):
    def __init__(self, input_width, output_width):
        super().__init__()
        self.linear = nn.Linear(input_width, output_width, bias=False)
        self.register_buffer('input_min', torch.tensor(-float('inf')))
        self.register_buffer('input_max', torch.tensor(float('inf')))
        self.register_buffer('output_min', torch.tensor(-float('inf')))
        self.register_buffer('output_max', torch.tensor(float('inf')))

    def forward(self, x):
        # Checkpoint clipping bounds default to unbounded for random weights.
        x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        return torch.clamp(x, self.output_min, self.output_max)


# %% [notebook-only]
example_model = ClippedLinear(16, 24)
example_outputs = example_model(torch.ones(1, 3, 16))
print('ClippedLinear: (1, 3, 16) ->', tuple(example_outputs.shape))


# %%
def rotary(x, positions, base):
    # Rotate contiguous half-head pairs: (batch, steps, heads, head_dim).
    half = x.shape[-1] // 2
    frequency = torch.arange(half, device=x.device, dtype=torch.float32)
    frequency = base ** (-frequency / half)
    angles = positions[..., None].float() * frequency
    angles = torch.cat((angles, angles), dim=-1)
    cosine = angles.cos().to(x.dtype)[:, :, None, :]
    sine = angles.sin().to(x.dtype)[:, :, None, :]
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cosine + rotated * sine


def gated_mlp(x, gate, up, down):
    gate_values = gate(x)
    gate_values = F.gelu(gate_values, approximate='tanh')
    up_values = up(x)
    mixed = gate_values * up_values
    return down(mixed)


# %%
class EncoderAttention(nn.Module):
    def __init__(self, width, heads, kv_heads, head_dim, base=10000.0, radius=None, axial=False):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.base, self.radius, self.axial = base, radius, axial
        self.q_proj = nn.Linear(width, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, width, bias=False)
        self.q_norm, self.k_norm = RMSNorm(head_dim), RMSNorm(head_dim)
        self.v_norm = RMSNorm(head_dim, scale=False)

    def forward(self, x, valid, positions):
        valid = valid.to(dtype=torch.bool)
        batch_size, steps, _ = x.shape
        # Project and normalize attention heads: (batch, steps, width) -> (batch, steps, heads, head_dim).
        # @arch eg.attention.qkv:start
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        query = query.reshape(batch_size, steps, self.heads, self.head_dim)
        key = key.reshape(batch_size, steps, self.kv_heads, self.head_dim)
        value = value.reshape(batch_size, steps, self.kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        value = self.v_norm(value)
        # @arch eg.attention.qkv:end

        # Encode 1D token positions or independent x/y patch axes in Q and K.
        # @arch eg.attention.rope:start
        if self.axial:
            query_axes = query.chunk(2, dim=-1)
            key_axes = key.chunk(2, dim=-1)
            query = torch.cat([rotary(query_axes[i], positions[..., i], self.base) for i in range(2)], dim=-1)
            key = torch.cat([rotary(key_axes[i], positions[..., i], self.base) for i in range(2)], dim=-1)
        else:
            query = rotary(query, positions, self.base)
            key = rotary(key, positions, self.base)
        # @arch eg.attention.rope:end

        # Mask padding and optionally restrict bidirectional distance: (batch, 1, steps, steps).
        # @arch eg.attention.mask:start
        allowed = valid[:, None, None, :].expand(batch_size, 1, steps, steps)
        if self.radius is not None:
            distance = positions[:, :, None] - positions[:, None, :]
            allowed = allowed & (distance.abs() <= self.radius)[:, None, :, :]
        # @arch eg.attention.mask:end

        # Q/K RMS normalization makes the released attention scale exactly 1.0.
        # Share K/V across query groups, mix values, and merge heads back to width.
        # @arch eg.attention.mix:start
        groups = self.heads // self.kv_heads
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)
        scores = torch.einsum('bthd,bshd->bhts', query, key)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        context = torch.einsum('bhts,bshd->bthd', weights, value)
        context = context.reshape(batch_size, steps, self.heads * self.head_dim)
        return self.o_proj(context)
        # @arch eg.attention.mix:end


# %% [notebook-only]
example_model = EncoderAttention(16, 4, 2, 4, radius=2)
example_outputs = example_model(torch.ones(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), torch.arange(4)[None])
print('EncoderAttention: (1, 4, 16) ->', tuple(example_outputs.shape))


# %%
class EncoderBlock(nn.Module):
    def __init__(self, width, hidden, heads, kv_heads, head_dim, base, radius=None, axial=False, ple_width=None):
        super().__init__()
        self.attention = EncoderAttention(width, heads, kv_heads, head_dim, base, radius, axial)
        self.input_norm, self.attention_norm = RMSNorm(width), RMSNorm(width)
        self.pre_ffn_norm, self.post_ffn_norm = RMSNorm(width), RMSNorm(width)
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
        self.ple_width = ple_width
        if ple_width is not None:
            self.ple_gate = nn.Linear(width, ple_width, bias=False)
            self.ple_projection = nn.Linear(ple_width, width, bias=False)
            self.ple_norm = RMSNorm(width)
            self.register_buffer('layer_scalar', torch.ones(1))

    def forward(self, x, valid, positions, per_layer_input=None):
        # Normalize the attention branch before adding its residual: (batch, steps, width).
        # @arch eg.block.attention:start
        normalized = self.input_norm(x)
        attended = self.attention(normalized, valid, positions)
        attended = self.attention_norm(attended)
        x = x + attended
        # @arch eg.block.attention:end
        # Multiply GELU gate and up branches, project down, normalize, and add the residual.
        # @arch eg.block.ffn:start
        normalized = self.pre_ffn_norm(x)
        feed_forward = gated_mlp(normalized, self.gate, self.up, self.down)
        feed_forward = self.post_ffn_norm(feed_forward)
        x = x + feed_forward
        # @arch eg.block.ffn:end
        # Gate this layer’s original-input projection into the residual stream.
        # @arch eg.block.ple:start
        if self.ple_width is not None:
            gate = self.ple_gate(x)
            gate = F.gelu(gate, approximate='tanh')
            gated_input = gate * per_layer_input
            projected = self.ple_projection(gated_input)
            projected = self.ple_norm(projected)
            x = x + projected
            x = x * self.layer_scalar
        return x
        # @arch eg.block.ple:end


# %% [notebook-only]
example_model = EncoderBlock(16, 32, 4, 2, 4, 10000.0, radius=2, ple_width=8)
example_outputs = example_model(torch.ones(1, 4, 16), torch.ones(1, 4, dtype=torch.bool), torch.arange(4)[None], torch.ones(1, 4, 8))
print('EncoderBlock: (1, 4, 16) ->', tuple(example_outputs.shape))


# %%
class TextBackbone(nn.Module):
    def __init__(self, config=TEXT_CONFIG):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab'], config['width'], padding_idx=0)
        self.ple = nn.Linear(config['width'], config['layers'] * config['ple_width'], bias=False)
        self.ple_norm = RMSNorm(config['ple_width'])
        self.layers = nn.ModuleList()
        for index in range(config['layers']):
            global_layer = (index + 1) % 6 == 0
            kv_heads = 1 if global_layer else config['local_kv']
            head_dim = config['global_dim'] if global_layer else config['local_dim']
            base = 1000000.0 if global_layer else 10000.0
            radius = None if global_layer else config['radius']
            self.layers.append(EncoderBlock(config['width'], config['hidden'], config['heads'], kv_heads,
                                            head_dim, base, radius, ple_width=config['ple_width']))
        self.norm = RMSNorm(config['width'])
        self.projection = nn.Linear(config['width'], config['output'], bias=False)

    def embed(self, input_ids):
        # Look up token vectors and scale by sqrt(width): (batch, steps) -> (batch, steps, width).
        # @arch eg.text.embedding:start
        embeddings = self.embedding(input_ids)
        return embeddings * (self.config['width'] ** 0.5)
        # @arch eg.text.embedding:end

    def forward(self, embeddings, valid, positions=None):
        valid = valid.to(dtype=torch.bool)
        batch_size, steps, _ = embeddings.shape
        if steps > 8192:
            raise ValueError('The supported input context is 8192 tokens.')
        if positions is None:
            positions = torch.arange(steps, device=embeddings.device)[None].expand(batch_size, steps)
        # Project original embeddings into normalized inputs for all layers: (batch, steps, layers, ple_width).
        # @arch eg.text.ple:start
        per_layer_projection = self.ple(embeddings)
        per_layer_projection = per_layer_projection * (self.config['width'] ** -0.5)
        per_layer_projection = per_layer_projection.reshape(batch_size, steps, self.config['layers'], self.config['ple_width'])
        per_layer_projection = self.ple_norm(per_layer_projection)
        # @arch eg.text.ple:end
        # Run the five-local/one-global sequence using each layer’s own input slice.
        # @arch eg.text.layers:start
        x = embeddings
        for index, layer in enumerate(self.layers):
            layer_input = per_layer_projection[:, :, index, :]
            x = layer(x, valid, positions, layer_input)
        # @arch eg.text.layers:end
        # Normalize and project each token: (batch, steps, width) -> (batch, steps, output).
        # @arch eg.text.output:start
        x = self.norm(x)
        return self.projection(x)
        # @arch eg.text.output:end


# %% [notebook-only]
example_model = TextBackbone(example_text_config)
example_embeddings = example_model.embed(torch.tensor([[1, 2, 3, 0]]))
example_outputs = example_model(example_embeddings, torch.tensor([[True, True, True, False]]))
print('TextBackbone: (1, 4) ->', tuple(example_outputs.shape))


# %%
class VisionEncoder(nn.Module):
    def __init__(self, config=VISION_CONFIG):
        super().__init__()
        self.config = config
        self.patch_projection = nn.Linear(3 * config['patch'] ** 2, config['width'], bias=False)
        self.position_table = nn.Parameter(torch.ones(2, config['positions'], config['width']))
        self.layers = nn.ModuleList([EncoderBlock(config['width'], config['hidden'], config['heads'],
                                    config['heads'], config['head_dim'], 100.0, axial=True)
                                    for _ in range(config['layers'])])

    def forward(self, patches, positions):
        # Consume flattened RGB patches and mark valid 2D patch positions.
        # @arch eg.vision.input:start
        valid = (positions >= 0).all(dim=-1)
        scaled_pixels = 2 * (patches - 0.5)
        # @arch eg.vision.input:end
        # Project patches and add learned x/y positions: (batch, patches, vision_width).
        # @arch eg.vision.patch:start
        x = self.patch_projection(scaled_pixels)
        coordinates = positions.clamp(min=0)
        x_positions = self.position_table[0][coordinates[..., 0]]
        y_positions = self.position_table[1][coordinates[..., 1]]
        positional = (x_positions + y_positions) * valid[..., None]
        x = x + positional
        # @arch eg.vision.patch:end
        # Encode patches with bidirectional axial attention and gated feed-forward blocks.
        # @arch eg.vision.layers:start
        for layer in self.layers:
            x = layer(x, valid, positions)
        # @arch eg.vision.layers:end
        # Spatial pooling follows processor coordinates, not consecutive patch order.
        # Pool by 2D coordinates and scale: (batch, patches, width) -> (batch, soft_tokens, width).
        # @arch eg.vision.pool:start
        kernel = self.config['pool']
        output_length = patches.shape[1] // (kernel * kernel)
        max_x = coordinates[..., 0].amax(dim=1, keepdim=True) + 1
        cells = coordinates // kernel
        cell_ids = cells[..., 0] + (max_x // kernel) * cells[..., 1]
        weights = F.one_hot(cell_ids, output_length).float() / (kernel * kernel)
        x = x.masked_fill(~valid[..., None], 0.0)
        pooled = torch.einsum('bpo,bpd->bod', weights, x.float())
        pooled = pooled * (self.config['width'] ** 0.5)
        pool_valid = torch.einsum('bpo,bp->bo', weights, valid.float()) > 0
        return pooled, pool_valid
        # @arch eg.vision.pool:end


# %% [notebook-only]
example_model = VisionEncoder(example_vision_config)
example_positions = torch.tensor([[[x, y] for y in range(4) for x in range(4)]])
example_patches = torch.rand(1, 16, 12)
example_outputs, example_mask = example_model(example_patches, example_positions)
print('VisionEncoder: (1, 16, 12) ->', tuple(example_outputs.shape), tuple(example_mask.shape))


# %%
class AudioAttention(nn.Module):
    def __init__(self, config=AUDIO_CONFIG):
        super().__init__()
        self.config = config
        width, heads = config['width'], config['heads']
        self.head_dim = width // heads
        self.q_proj, self.k_proj = ClippedLinear(width, width), ClippedLinear(width, width)
        self.v_proj, self.post = ClippedLinear(width, width), ClippedLinear(width, width)
        self.relative_k = nn.Linear(width, width, bias=False)
        self.per_dim_scale = nn.Parameter(torch.zeros(self.head_dim))

    def forward(self, x, valid):
        valid = valid.to(dtype=torch.bool)
        batch_size, steps, width = x.shape
        heads, dim = self.config['heads'], self.head_dim
        chunk, past, future = self.config['chunk'], self.config['left'] - 1, self.config['right']
        context = chunk + past + future
        blocks = (steps + chunk - 1) // chunk
        # Project audio Q/K/V and apply learned query scaling: (batch, steps, heads, head_dim).
        # @arch eg.audio.qkv:start
        query = self.q_proj(x).float().reshape(batch_size, steps, heads, dim)
        key = self.k_proj(x).float().reshape(batch_size, steps, heads, dim)
        value = self.v_proj(x).float().reshape(batch_size, steps, heads, dim)
        query = query * (dim ** -0.5 / math.log(2)) * F.softplus(self.per_dim_scale)
        key = key * (math.log(1 + math.e) / math.log(2))
        # @arch eg.audio.qkv:end
        # Project descending sinusoidal relative positions into per-head keys.
        # @arch eg.audio.relative:start
        timescales = torch.arange(width // 2, device=x.device, dtype=torch.float32)
        timescales = torch.exp(timescales * (-math.log(10000.0) / max(width // 2 - 1, 1)))
        position_offsets = torch.arange(context // 2, -1, -1, device=x.device)
        angles = position_offsets[:, None] * timescales
        relative = torch.cat((angles.sin(), angles.cos()), dim=-1).to(x.dtype)
        relative_key = self.relative_k(relative).reshape(-1, heads, dim).float()
        # @arch eg.audio.relative:end
        # Gather overlapping context and shift relative logits: (batch, heads, blocks, chunk, context).
        # @arch eg.audio.blocks:start
        query = F.pad(query, (0, 0, 0, 0, 0, blocks * chunk - steps))
        query = query.reshape(batch_size, blocks, chunk, heads, dim)
        context_ids = torch.arange(blocks, device=x.device)[:, None] * chunk
        context_ids = context_ids + torch.arange(context, device=x.device)[None] - past
        safe_ids = context_ids.clamp(0, steps - 1)
        in_bounds = (context_ids >= 0) & (context_ids < steps)
        key = key[:, safe_ids] * in_bounds[None, :, :, None, None]
        value = value[:, safe_ids] * in_bounds[None, :, :, None, None]
        content_scores = torch.einsum('bnqhd,bnkhd->bhnqk', query, key)
        relative_scores = torch.einsum('bnqhd,rhd->bhnqr', query, relative_key)
        relative_scores = F.pad(relative_scores, (0, context + 1 - relative_key.shape[0]))
        relative_scores = relative_scores.reshape(batch_size, heads, blocks, chunk * (context + 1))
        relative_scores = relative_scores[..., :chunk * context]
        relative_scores = relative_scores.reshape(batch_size, heads, blocks, chunk, context)
        scores = content_scores + relative_scores
        # @arch eg.audio.blocks:end
        # Apply context and padding masks, cap logits, softmax, and mix audio values.
        # @arch eg.audio.mask:start
        query_ids = torch.arange(blocks * chunk, device=x.device).reshape(blocks, chunk)
        distance = query_ids[:, :, None] - context_ids[:, None, :]
        allowed = (distance >= 0) & (distance < past)
        allowed = allowed | ((distance < 0) & (-distance < future))
        allowed = allowed & in_bounds[:, None, :] & (query_ids < steps)[:, :, None]
        allowed = allowed[None] & valid[:, safe_ids][:, :, None, :]
        scores = 50.0 * torch.tanh(scores / 50.0)
        scores = scores.masked_fill(~allowed[:, None], -1e9)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum('bhnqk,bnkhd->bnqhd', weights, value)
        attended = attended.reshape(batch_size, blocks * chunk, width)[:, :steps]
        return self.post(attended.to(x.dtype))
        # @arch eg.audio.mask:end


# %% [notebook-only]
example_model = AudioAttention(example_audio_config)
example_outputs = example_model(torch.ones(1, 7, 16), torch.ones(1, 7, dtype=torch.bool))
print('AudioAttention: (1, 7, 16) ->', tuple(example_outputs.shape))


# %%
class AudioFeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.pre_norm, self.post_norm = RMSNorm(width), RMSNorm(width)
        self.first, self.second = ClippedLinear(width, 4 * width), ClippedLinear(4 * width, width)

    def forward(self, x):
        # Apply the SiLU feed-forward branch with a half-step residual: (batch, steps, width).
        # @arch eg.audio.ffn:start
        residual = x
        clipped = torch.clamp(x, -1e10, 1e10)
        normalized = self.pre_norm(clipped)
        expanded = self.first(normalized)
        activated = F.silu(expanded)
        projected = self.second(activated)
        projected = torch.clamp(projected, -1e10, 1e10)
        projected = self.post_norm(projected)
        return residual + 0.5 * projected
        # @arch eg.audio.ffn:end


# %% [notebook-only]
example_model = AudioFeedForward(16)
example_outputs = example_model(torch.ones(1, 7, 16))
print('AudioFeedForward: (1, 7, 16) ->', tuple(example_outputs.shape))


# %%
class AudioBlock(nn.Module):
    def __init__(self, config=AUDIO_CONFIG):
        super().__init__()
        width = config['width']
        self.config = config
        self.ffn1, self.ffn2 = AudioFeedForward(width), AudioFeedForward(width)
        self.attention = AudioAttention(config)
        self.pre_attention, self.post_attention = RMSNorm(width), RMSNorm(width)
        self.pre_conv, self.conv_norm, self.output_norm = RMSNorm(width), RMSNorm(width), RMSNorm(width)
        self.conv_start, self.conv_end = ClippedLinear(width, 2 * width), ClippedLinear(width, width)
        self.depthwise = nn.Conv1d(width, width, config['kernel'], groups=width, bias=False)

    def forward(self, x, valid):
        # Apply the first half-step feed-forward, normalized attention, and residual.
        # @arch eg.audio.block_attention:start
        x = self.ffn1(x)
        normalized = self.pre_attention(torch.clamp(x, -1e10, 1e10))
        attended = self.attention(normalized, valid)
        attended = self.post_attention(torch.clamp(attended, -1e10, 1e10))
        x = x + attended
        # @arch eg.audio.block_attention:end
        # Use a GLU gate and left-padded depthwise convolution without future context.
        # @arch eg.audio.conv:start
        normalized = self.pre_conv(x)
        projected = self.conv_start(normalized)
        signal, gate = projected.chunk(2, dim=-1)
        signal = signal * torch.sigmoid(gate)
        channels_first = signal.transpose(1, 2)
        padded = F.pad(channels_first, (self.config['kernel'] - 1, 0))
        convolved = self.depthwise(padded).transpose(1, 2)
        convolved = self.conv_norm(torch.clamp(convolved, -1e10, 1e10))
        activated = F.silu(convolved)
        x = x + self.conv_end(activated)
        # @arch eg.audio.conv:end
        # Apply the second half-step feed-forward and final block normalization.
        # @arch eg.audio.block_output:start
        x = self.ffn2(x)
        return self.output_norm(torch.clamp(x, -1e10, 1e10))
        # @arch eg.audio.block_output:end


# %% [notebook-only]
example_model = AudioBlock(example_audio_config)
example_outputs = example_model(torch.ones(1, 7, 16), torch.ones(1, 7, dtype=torch.bool))
print('AudioBlock: (1, 7, 16) ->', tuple(example_outputs.shape))


# %%
class AudioEncoder(nn.Module):
    def __init__(self, config=AUDIO_CONFIG):
        super().__init__()
        self.config = config
        first, second = config['channels']
        self.convs = nn.ModuleList([nn.Conv2d(1, first, 3, stride=2, padding=1, bias=False),
                                    nn.Conv2d(first, second, 3, stride=2, padding=1, bias=False)])
        self.norms = nn.ModuleList([nn.LayerNorm(first, eps=1e-6, bias=False),
                                    nn.LayerNorm(second, eps=1e-6, bias=False)])
        self.input_projection = nn.Linear(((config['bins'] + 3) // 4) * second, config['width'], bias=False)
        self.layers = nn.ModuleList([AudioBlock(config) for _ in range(config['layers'])])
        self.output_projection = nn.Linear(config['width'], config['output'], bias=True)

    def forward(self, features, valid):
        valid = valid.to(dtype=torch.bool)
        # Consume audio features: (batch, frames, mel_bins).
        # @arch eg.audio.input:start
        x = features[:, None]
        # @arch eg.audio.input:end
        # Subsample time and frequency twice, then project into audio hidden width.
        # @arch eg.audio.subsample:start
        for conv, norm in zip(self.convs, self.norms):
            x = x * valid[:, None, :, None]
            x = conv(x)
            channels_last = x.permute(0, 2, 3, 1)
            normalized = norm(channels_last)
            x = F.relu(normalized).permute(0, 3, 1, 2)
            valid = valid[:, ::2]
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.input_projection(x)
        # @arch eg.audio.subsample:end
        # Encode subsampled audio with relative-attention Conformer blocks.
        # @arch eg.audio.layers:start
        for layer in self.layers:
            x = layer(x, valid)
        # @arch eg.audio.layers:end
        # Project audio hidden states into modality output width: (batch, soft_tokens, audio_output).
        # @arch eg.audio.output:start
        return self.output_projection(x), valid
        # @arch eg.audio.output:end


# %% [notebook-only]
example_model = AudioEncoder(example_audio_config)
example_features = torch.randn(1, 25, 8)
example_feature_mask = torch.ones(1, 25, dtype=torch.bool)
example_outputs, example_mask = example_model(example_features, example_feature_mask)
print('AudioEncoder: (1, 25, 8) ->', tuple(example_outputs.shape), tuple(example_mask.shape))


# %%
class ModalityProjection(nn.Module):
    def __init__(self, input_width, text_width):
        super().__init__()
        self.norm = RMSNorm(input_width, scale=False)
        self.projection = nn.Linear(input_width, text_width, bias=False)

    def forward(self, x):
        # Normalize without a learned scale and project soft tokens to text width.
        # @arch eg.modality.project:start
        normalized = self.norm(x)
        return self.projection(normalized)
        # @arch eg.modality.project:end


# %% [notebook-only]
example_model = ModalityProjection(24, 16)
example_outputs = example_model(torch.ones(1, 7, 24))
print('ModalityProjection: (1, 7, 24) ->', tuple(example_outputs.shape))


# %%
def merge_soft_tokens(embeddings, input_ids, token_id, soft_tokens, valid):
    # Validate and pack per-example soft tokens into their interleaved placeholder slots.
    slots = input_ids == token_id
    if not torch.equal(slots.sum(dim=1), valid.sum(dim=1)):
        raise ValueError('Each example must have one placeholder per valid soft token.')
    feature_order = torch.argsort(~valid, dim=1, stable=True)
    packed = soft_tokens.gather(1, feature_order[..., None].expand_as(soft_tokens))
    rank = slots.long().cumsum(dim=1) - 1
    rank = rank.clamp(min=0)
    gathered = packed.gather(1, rank[..., None].expand(-1, -1, embeddings.shape[-1]))
    return torch.where(slots[..., None], gathered, embeddings)


# %%
class EmbeddingGemma2(nn.Module):
    def __init__(self, text=TEXT_CONFIG, vision=VISION_CONFIG, audio=AUDIO_CONFIG):
        super().__init__()
        self.text = TextBackbone(text)
        self.vision = VisionEncoder(vision) if vision is not None else None
        self.audio = AudioEncoder(audio) if audio is not None else None
        self.embed_vision = ModalityProjection(vision['width'], text['width']) if vision is not None else None
        self.embed_audio = ModalityProjection(audio['output'], text['width']) if audio is not None else None

    def forward(self, input_ids, valid, patches=None, patch_positions=None,
                features=None, feature_valid=None, output_dim=None, video=False):
        valid = valid.to(dtype=torch.bool)
        # Consume token IDs, including modality placeholder slots.
        # @arch eg.input:start
        embeddings = self.text.embed(input_ids)
        # @arch eg.input:end
        # Encode images or independent video frames and replace vision placeholder embeddings.
        # @arch eg.merge.vision:start
        if patches is not None:
            if self.vision is None or patch_positions is None:
                raise ValueError('Vision encoder and patch positions are required.')
            if patches.ndim == 4:
                batch_size, media_count, patch_count, patch_width = patches.shape
                flat_patches = patches.reshape(batch_size * media_count, patch_count, patch_width)
                flat_positions = patch_positions.reshape(batch_size * media_count, patch_count, 2)
                vision_tokens, vision_valid = self.vision(flat_patches, flat_positions)
                vision_tokens = vision_tokens.reshape(batch_size, -1, vision_tokens.shape[-1])
                vision_valid = vision_valid.reshape(batch_size, -1)
            else:
                vision_tokens, vision_valid = self.vision(patches, patch_positions)
            vision_tokens = self.embed_vision(vision_tokens)
            placeholder = self.text.config['video_token'] if video else self.text.config['image_token']
            embeddings = merge_soft_tokens(embeddings, input_ids, placeholder, vision_tokens, vision_valid)
        # @arch eg.merge.vision:end
        # Encode audio and replace placeholders with valid projected audio tokens.
        # @arch eg.merge.audio:start
        if features is not None:
            if self.audio is None or feature_valid is None:
                raise ValueError('Audio encoder and feature mask are required.')
            audio_tokens, audio_valid = self.audio(features, feature_valid)
            audio_tokens = self.embed_audio(audio_tokens)
            embeddings = merge_soft_tokens(embeddings, input_ids, self.text.config['audio_token'], audio_tokens, audio_valid)
        # @arch eg.merge.audio:end
        # Average projected token vectors over valid positions: (batch, steps, output) -> (batch, output).
        # @arch eg.pool:start
        token_outputs = self.text(embeddings, valid)
        masked = token_outputs * valid[..., None]
        count = valid.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = masked.sum(dim=1) / count
        # @arch eg.pool:end
        # Keep leading Matryoshka dimensions, then normalize: (batch, output_dim).
        # @arch eg.output:start
        dimension = token_outputs.shape[-1] if output_dim is None else output_dim
        if dimension <= 0 or dimension > token_outputs.shape[-1]:
            raise ValueError('Output dimension must fit the native embedding width.')
        truncated = pooled[:, :dimension]
        return F.normalize(truncated, p=2, dim=-1, eps=1e-12)
        # @arch eg.output:end


# %% [notebook-only]
example_model = EmbeddingGemma2(example_text_config, vision=None, audio=None)
example_ids = torch.tensor([[1, 2, 3, 0]])
example_outputs = example_model(example_ids, example_ids != 0)
print('EmbeddingGemma2 text: (1, 4) ->', tuple(example_outputs.shape))


# %% [notebook-only]
example_model = EmbeddingGemma2(example_text_config, example_vision_config, example_audio_config)
example_ids = torch.tensor([[1, 61, 61, 61, 61, 2, 62, 62, 62, 62, 62, 62, 62, 3, 0]])
example_positions = torch.tensor([[[x, y] for y in range(4) for x in range(4)]])
example_outputs = example_model(example_ids, example_ids != 0, torch.rand(1, 16, 12),
                                example_positions, torch.randn(1, 25, 8),
                                torch.ones(1, 25, dtype=torch.bool), output_dim=6)
print('EmbeddingGemma2 text + image + audio: (1, 15) ->', tuple(example_outputs.shape),
      'L2 norm:', example_outputs.norm(dim=-1).detach())
