# %%
import math
import jax
import jax.numpy as jnp
from flax import linen as nn


# %% [markdown]
# # EmbeddingGemma 2 in JAX / Flax
# Architecture implementation with random weights, translated from the released PyTorch source.
# The trained checkpoint is not loaded. Tokenization and media feature extraction are external.
# Inputs: token IDs, RGB patches in [0, 1] with (x, y) coordinates, and 128-bin audio features.
# Video frames share the vision encoder. Defaults match the checkpoint; smoke tests use small widths.
# Use float32 or bfloat16. Source provenance and boundaries: docs/embeddinggemma2.md.


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
    width: int
    scale: bool = True

    @nn.compact
    def __call__(self, x):
        # @arch eg.norm:start
        values = x.astype(jnp.float32)
        mean_square = jnp.mean(jnp.square(values), axis=-1, keepdims=True)
        normalized = values * jnp.power(mean_square + 1e-6, -0.5)
        if self.scale:
            weight = self.param('weight', nn.initializers.ones, (self.width,))
            normalized = normalized * weight
        return normalized.astype(x.dtype)
        # @arch eg.norm:end


# %% [notebook-only]
example_model = RMSNorm(16)
example_inputs = jnp.ones((1, 3, 16))
example_params = example_model.init(jax.random.key(0), example_inputs)
example_outputs = example_model.apply(example_params, example_inputs)
print('RMSNorm: (1, 3, 16) ->', example_outputs.shape)


# %%
class ClippedLinear(nn.Module):
    input_width: int
    output_width: int

    @nn.compact
    def __call__(self, x):
        input_min = self.variable('bounds', 'input_min', lambda: jnp.array(-jnp.inf))
        input_max = self.variable('bounds', 'input_max', lambda: jnp.array(jnp.inf))
        output_min = self.variable('bounds', 'output_min', lambda: jnp.array(-jnp.inf))
        output_max = self.variable('bounds', 'output_max', lambda: jnp.array(jnp.inf))
        x = jnp.clip(x, input_min.value, input_max.value)
        x = nn.Dense(self.output_width, use_bias=False, name='linear')(x)
        return jnp.clip(x, output_min.value, output_max.value)


# %% [notebook-only]
example_model = ClippedLinear(16, 24)
example_inputs = jnp.ones((1, 3, 16))
example_params = example_model.init(jax.random.key(0), example_inputs)
example_outputs = example_model.apply(example_params, example_inputs)
print('ClippedLinear: (1, 3, 16) ->', example_outputs.shape)


# %%
def rotary(x, positions, base):
    half = x.shape[-1] // 2
    frequency = jnp.arange(half, dtype=jnp.float32)
    frequency = base ** (-frequency / half)
    angles = positions[..., None].astype(jnp.float32) * frequency
    angles = jnp.concatenate((angles, angles), axis=-1)
    cosine = jnp.cos(angles).astype(x.dtype)[:, :, None, :]
    sine = jnp.sin(angles).astype(x.dtype)[:, :, None, :]
    rotated = jnp.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cosine + rotated * sine


def gated_mlp(x, gate, up, down):
    gate_values = gate(x)
    gate_values = jax.nn.gelu(gate_values, approximate=True)
    up_values = up(x)
    mixed = gate_values * up_values
    return down(mixed)


# %%
class EncoderAttention(nn.Module):
    width: int
    heads: int
    kv_heads: int
    head_dim: int
    base: float = 10000.0
    radius: int | None = None
    axial: bool = False

    @nn.compact
    def __call__(self, x, valid, positions):
        valid = valid.astype(jnp.bool_)
        batch_size, steps, _ = x.shape
        # Project and normalize attention heads: (batch, steps, width) -> (batch, steps, heads, head_dim).
        # @arch eg.attention.qkv:start
        query = nn.Dense(self.heads * self.head_dim, use_bias=False, name='q_proj')(x)
        key = nn.Dense(self.kv_heads * self.head_dim, use_bias=False, name='k_proj')(x)
        value = nn.Dense(self.kv_heads * self.head_dim, use_bias=False, name='v_proj')(x)
        query = query.reshape(batch_size, steps, self.heads, self.head_dim)
        key = key.reshape(batch_size, steps, self.kv_heads, self.head_dim)
        value = value.reshape(batch_size, steps, self.kv_heads, self.head_dim)
        query = RMSNorm(self.head_dim, name='q_norm')(query)
        key = RMSNorm(self.head_dim, name='k_norm')(key)
        value = RMSNorm(self.head_dim, scale=False, name='v_norm')(value)
        # @arch eg.attention.qkv:end
        # Encode 1D token positions or independent x/y patch axes in Q and K.
        # @arch eg.attention.rope:start
        if self.axial:
            query_axes = jnp.split(query, 2, axis=-1)
            key_axes = jnp.split(key, 2, axis=-1)
            query = jnp.concatenate([rotary(query_axes[i], positions[..., i], self.base) for i in range(2)], axis=-1)
            key = jnp.concatenate([rotary(key_axes[i], positions[..., i], self.base) for i in range(2)], axis=-1)
        else:
            query = rotary(query, positions, self.base)
            key = rotary(key, positions, self.base)
        # @arch eg.attention.rope:end
        # Mask padding and optionally restrict bidirectional distance: (batch, 1, steps, steps).
        # @arch eg.attention.mask:start
        allowed = jnp.broadcast_to(valid[:, None, None, :], (batch_size, 1, steps, steps))
        if self.radius is not None:
            distance = positions[:, :, None] - positions[:, None, :]
            allowed = allowed & (jnp.abs(distance) <= self.radius)[:, None, :, :]
        # @arch eg.attention.mask:end
        # Share K/V across query groups, mix values, and merge heads back to width.
        # @arch eg.attention.mix:start
        groups = self.heads // self.kv_heads
        key = jnp.repeat(key, groups, axis=2)
        value = jnp.repeat(value, groups, axis=2)
        scores = jnp.einsum('bthd,bshd->bhts', query, key)
        scores = jnp.where(allowed, scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(value.dtype)
        context = jnp.einsum('bhts,bshd->bthd', weights, value)
        context = context.reshape(batch_size, steps, self.heads * self.head_dim)
        return nn.Dense(self.width, use_bias=False, name='o_proj')(context)
        # @arch eg.attention.mix:end


# %% [notebook-only]
example_model = EncoderAttention(16, 4, 2, 4, radius=2)
example_inputs = (jnp.ones((1, 4, 16)), jnp.ones((1, 4), dtype=bool), jnp.arange(4)[None])
example_params = example_model.init(jax.random.key(0), *example_inputs)
example_outputs = example_model.apply(example_params, *example_inputs)
print('EncoderAttention: (1, 4, 16) ->', example_outputs.shape)


# %%
class EncoderBlock(nn.Module):
    width: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    base: float
    radius: int | None = None
    axial: bool = False
    ple_width: int | None = None

    @nn.compact
    def __call__(self, x, valid, positions, per_layer_input=None):
        # Normalize the attention branch before adding its residual: (batch, steps, width).
        # @arch eg.block.attention:start
        normalized = RMSNorm(self.width, name='input_norm')(x)
        attended = EncoderAttention(self.width, self.heads, self.kv_heads, self.head_dim,
                                    self.base, self.radius, self.axial, name='attention')(normalized, valid, positions)
        attended = RMSNorm(self.width, name='attention_norm')(attended)
        x = x + attended
        # @arch eg.block.attention:end
        # Multiply GELU gate and up branches, project down, normalize, and add the residual.
        # @arch eg.block.ffn:start
        normalized = RMSNorm(self.width, name='pre_ffn_norm')(x)
        gate = nn.Dense(self.hidden, use_bias=False, name='gate')
        up = nn.Dense(self.hidden, use_bias=False, name='up')
        down = nn.Dense(self.width, use_bias=False, name='down')
        feed_forward = gated_mlp(normalized, gate, up, down)
        feed_forward = RMSNorm(self.width, name='post_ffn_norm')(feed_forward)
        x = x + feed_forward
        # @arch eg.block.ffn:end
        # Gate this layer’s original-input projection into the residual stream.
        # @arch eg.block.ple:start
        if self.ple_width is not None:
            gate = nn.Dense(self.ple_width, use_bias=False, name='ple_gate')(x)
            gate = jax.nn.gelu(gate, approximate=True)
            gated_input = gate * per_layer_input
            projected = nn.Dense(self.width, use_bias=False, name='ple_projection')(gated_input)
            projected = RMSNorm(self.width, name='ple_norm')(projected)
            x = x + projected
            layer_scalar = self.variable('constants', 'layer_scalar', lambda: jnp.ones((1,)))
            x = x * layer_scalar.value
        return x
        # @arch eg.block.ple:end


# %% [notebook-only]
example_model = EncoderBlock(16, 32, 4, 2, 4, 10000.0, radius=2, ple_width=8)
example_inputs = (jnp.ones((1, 4, 16)), jnp.ones((1, 4), dtype=bool), jnp.arange(4)[None], jnp.ones((1, 4, 8)))
example_params = example_model.init(jax.random.key(0), *example_inputs)
example_outputs = example_model.apply(example_params, *example_inputs)
print('EncoderBlock: (1, 4, 16) ->', example_outputs.shape)


# %%
class TextBackbone(nn.Module):
    config: dict

    def setup(self):
        config = self.config
        self.embedding = nn.Embed(config['vocab'], config['width'])
        self.ple = nn.Dense(config['layers'] * config['ple_width'], use_bias=False)
        self.ple_norm = RMSNorm(config['ple_width'])
        layers = []
        for index in range(config['layers']):
            global_layer = (index + 1) % 6 == 0
            kv_heads = 1 if global_layer else config['local_kv']
            head_dim = config['global_dim'] if global_layer else config['local_dim']
            base = 1000000.0 if global_layer else 10000.0
            radius = None if global_layer else config['radius']
            layers.append(EncoderBlock(config['width'], config['hidden'], config['heads'], kv_heads,
                                       head_dim, base, radius, ple_width=config['ple_width'], name=f'layers_{index}'))
        self.layers = layers
        self.norm = RMSNorm(config['width'])
        self.projection = nn.Dense(config['output'], use_bias=False)

    def embed(self, input_ids):
        # Look up token vectors and scale by sqrt(width): (batch, steps) -> (batch, steps, width).
        # @arch eg.text.embedding:start
        embeddings = self.embedding(input_ids)
        return embeddings * (self.config['width'] ** 0.5)
        # @arch eg.text.embedding:end

    def __call__(self, embeddings, valid, positions=None):
        valid = valid.astype(jnp.bool_)
        batch_size, steps, _ = embeddings.shape
        if steps > 8192:
            raise ValueError('The supported input context is 8192 tokens.')
        if positions is None:
            positions = jnp.broadcast_to(jnp.arange(steps)[None], (batch_size, steps))
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
example_inputs = (jnp.ones((1, 4, 16)), jnp.array([[True, True, True, False]]))
example_params = example_model.init(jax.random.key(0), *example_inputs)
example_outputs = example_model.apply(example_params, *example_inputs)
print('TextBackbone: (1, 4, 16) ->', example_outputs.shape)


# %%
class VisionEncoder(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, patches, positions):
        config = self.config
        # Consume flattened RGB patches and mark valid 2D patch positions.
        # @arch eg.vision.input:start
        valid = jnp.all(positions >= 0, axis=-1)
        scaled_pixels = 2 * (patches - 0.5)
        # @arch eg.vision.input:end
        # Project patches and add learned x/y positions: (batch, patches, vision_width).
        # @arch eg.vision.patch:start
        x = nn.Dense(config['width'], use_bias=False, name='patch_projection')(scaled_pixels)
        position_table = self.param('position_table', nn.initializers.ones, (2, config['positions'], config['width']))
        coordinates = jnp.maximum(positions, 0)
        x_positions = position_table[0][coordinates[..., 0]]
        y_positions = position_table[1][coordinates[..., 1]]
        positional = (x_positions + y_positions) * valid[..., None]
        x = x + positional
        # @arch eg.vision.patch:end
        # Encode patches with bidirectional axial attention and gated feed-forward blocks.
        # @arch eg.vision.layers:start
        for index in range(config['layers']):
            x = EncoderBlock(config['width'], config['hidden'], config['heads'], config['heads'],
                             config['head_dim'], 100.0, axial=True, name=f'layers_{index}')(x, valid, positions)
        # @arch eg.vision.layers:end
        # Pool by 2D coordinates and scale: (batch, patches, width) -> (batch, soft_tokens, width).
        # @arch eg.vision.pool:start
        kernel = config['pool']
        output_length = patches.shape[1] // (kernel * kernel)
        max_x = jnp.max(coordinates[..., 0], axis=1, keepdims=True) + 1
        cells = coordinates // kernel
        cell_ids = cells[..., 0] + (max_x // kernel) * cells[..., 1]
        weights = jax.nn.one_hot(cell_ids, output_length) / (kernel * kernel)
        x = jnp.where(valid[..., None], x, 0.0)
        pooled = jnp.einsum('bpo,bpd->bod', weights, x.astype(jnp.float32))
        pooled = pooled * (config['width'] ** 0.5)
        pool_valid = jnp.einsum('bpo,bp->bo', weights, valid.astype(jnp.float32)) > 0
        return pooled, pool_valid
        # @arch eg.vision.pool:end


# %% [notebook-only]
example_model = VisionEncoder(example_vision_config)
example_positions = jnp.array([[[x, y] for y in range(4) for x in range(4)]])
example_inputs = (jnp.ones((1, 16, 12)) * 0.5, example_positions)
example_params = example_model.init(jax.random.key(0), *example_inputs)
example_outputs, example_mask = example_model.apply(example_params, *example_inputs)
print('VisionEncoder: (1, 16, 12) ->', example_outputs.shape, example_mask.shape)


# %%
class AudioAttention(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, valid):
        valid = valid.astype(jnp.bool_)
        batch_size, steps, width = x.shape
        heads, dim = self.config['heads'], self.config['width'] // self.config['heads']
        chunk, past, future = self.config['chunk'], self.config['left'] - 1, self.config['right']
        context = chunk + past + future
        blocks = (steps + chunk - 1) // chunk
        # Project audio Q/K/V and apply learned query scaling: (batch, steps, heads, head_dim).
        # @arch eg.audio.qkv:start
        query = ClippedLinear(width, width, name='q_proj')(x).astype(jnp.float32)
        key = ClippedLinear(width, width, name='k_proj')(x).astype(jnp.float32)
        value = ClippedLinear(width, width, name='v_proj')(x).astype(jnp.float32)
        query = query.reshape(batch_size, steps, heads, dim)
        key = key.reshape(batch_size, steps, heads, dim)
        value = value.reshape(batch_size, steps, heads, dim)
        per_dim_scale = self.param('per_dim_scale', nn.initializers.zeros, (dim,))
        query = query * (dim ** -0.5 / math.log(2)) * jax.nn.softplus(per_dim_scale)
        key = key * (math.log(1 + math.e) / math.log(2))
        # @arch eg.audio.qkv:end
        # Project descending sinusoidal relative positions into per-head keys.
        # @arch eg.audio.relative:start
        timescales = jnp.arange(width // 2, dtype=jnp.float32)
        timescales = jnp.exp(timescales * (-math.log(10000.0) / max(width // 2 - 1, 1)))
        position_offsets = jnp.arange(context // 2, -1, -1)
        angles = position_offsets[:, None] * timescales
        relative = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1).astype(x.dtype)
        relative_key = nn.Dense(width, use_bias=False, name='relative_k')(relative)
        relative_key = relative_key.reshape(-1, heads, dim).astype(jnp.float32)
        # @arch eg.audio.relative:end
        # Gather overlapping context and shift relative logits: (batch, heads, blocks, chunk, context).
        # @arch eg.audio.blocks:start
        query = jnp.pad(query, ((0, 0), (0, blocks * chunk - steps), (0, 0), (0, 0)))
        query = query.reshape(batch_size, blocks, chunk, heads, dim)
        context_ids = jnp.arange(blocks)[:, None] * chunk
        context_ids = context_ids + jnp.arange(context)[None] - past
        safe_ids = jnp.clip(context_ids, 0, steps - 1)
        in_bounds = (context_ids >= 0) & (context_ids < steps)
        key = key[:, safe_ids] * in_bounds[None, :, :, None, None]
        value = value[:, safe_ids] * in_bounds[None, :, :, None, None]
        content_scores = jnp.einsum('bnqhd,bnkhd->bhnqk', query, key)
        relative_scores = jnp.einsum('bnqhd,rhd->bhnqr', query, relative_key)
        relative_scores = jnp.pad(relative_scores, ((0, 0), (0, 0), (0, 0), (0, 0), (0, context + 1 - relative_key.shape[0])))
        relative_scores = relative_scores.reshape(batch_size, heads, blocks, chunk * (context + 1))
        relative_scores = relative_scores[..., :chunk * context]
        relative_scores = relative_scores.reshape(batch_size, heads, blocks, chunk, context)
        scores = content_scores + relative_scores
        # @arch eg.audio.blocks:end
        # Apply context and padding masks, cap logits, softmax, and mix audio values.
        # @arch eg.audio.mask:start
        query_ids = jnp.arange(blocks * chunk).reshape(blocks, chunk)
        distance = query_ids[:, :, None] - context_ids[:, None, :]
        allowed = (distance >= 0) & (distance < past)
        allowed = allowed | ((distance < 0) & (-distance < future))
        allowed = allowed & in_bounds[:, None, :] & (query_ids < steps)[:, :, None]
        allowed = allowed[None] & valid[:, safe_ids][:, :, None, :]
        scores = 50.0 * jnp.tanh(scores / 50.0)
        scores = jnp.where(allowed[:, None], scores, -1e9)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum('bhnqk,bnkhd->bnqhd', weights, value)
        attended = attended.reshape(batch_size, blocks * chunk, width)[:, :steps]
        return ClippedLinear(width, width, name='post')(attended.astype(x.dtype))
        # @arch eg.audio.mask:end


# %% [notebook-only]
example_model = AudioAttention(example_audio_config)
example_inputs = (jnp.ones((1, 7, 16)), jnp.ones((1, 7), dtype=bool))
example_params = example_model.init(jax.random.key(0), *example_inputs)
example_outputs = example_model.apply(example_params, *example_inputs)
print('AudioAttention: (1, 7, 16) ->', example_outputs.shape)


# %%
class AudioFeedForward(nn.Module):
    width: int

    @nn.compact
    def __call__(self, x):
        # Apply the SiLU feed-forward branch with a half-step residual: (batch, steps, width).
        # @arch eg.audio.ffn:start
        residual = x
        clipped = jnp.clip(x, -1e10, 1e10)
        normalized = RMSNorm(self.width, name='pre_norm')(clipped)
        expanded = ClippedLinear(self.width, 4 * self.width, name='first')(normalized)
        activated = jax.nn.silu(expanded)
        projected = ClippedLinear(4 * self.width, self.width, name='second')(activated)
        projected = jnp.clip(projected, -1e10, 1e10)
        projected = RMSNorm(self.width, name='post_norm')(projected)
        return residual + 0.5 * projected
        # @arch eg.audio.ffn:end


# %% [notebook-only]
example_model = AudioFeedForward(16)
example_inputs = jnp.ones((1, 7, 16))
example_params = example_model.init(jax.random.key(0), example_inputs)
example_outputs = example_model.apply(example_params, example_inputs)
print('AudioFeedForward: (1, 7, 16) ->', example_outputs.shape)


# %%
class AudioBlock(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, x, valid):
        width = self.config['width']
        # Apply the first half-step feed-forward, normalized attention, and residual.
        # @arch eg.audio.block_attention:start
        x = AudioFeedForward(width, name='ffn1')(x)
        normalized = RMSNorm(width, name='pre_attention')(jnp.clip(x, -1e10, 1e10))
        attended = AudioAttention(self.config, name='attention')(normalized, valid)
        attended = RMSNorm(width, name='post_attention')(jnp.clip(attended, -1e10, 1e10))
        x = x + attended
        # @arch eg.audio.block_attention:end
        # Use a GLU gate and left-padded depthwise convolution without future context.
        # @arch eg.audio.conv:start
        normalized = RMSNorm(width, name='pre_conv')(x)
        projected = ClippedLinear(width, 2 * width, name='conv_start')(normalized)
        signal, gate = jnp.split(projected, 2, axis=-1)
        signal = signal * jax.nn.sigmoid(gate)
        convolved = nn.Conv(width, (self.config['kernel'],), padding=((self.config['kernel'] - 1, 0),),
                             feature_group_count=width, use_bias=False, name='depthwise')(signal)
        convolved = RMSNorm(width, name='conv_norm')(jnp.clip(convolved, -1e10, 1e10))
        activated = jax.nn.silu(convolved)
        x = x + ClippedLinear(width, width, name='conv_end')(activated)
        # @arch eg.audio.conv:end
        # Apply the second half-step feed-forward and final block normalization.
        # @arch eg.audio.block_output:start
        x = AudioFeedForward(width, name='ffn2')(x)
        return RMSNorm(width, name='output_norm')(jnp.clip(x, -1e10, 1e10))
        # @arch eg.audio.block_output:end


# %% [notebook-only]
example_model = AudioBlock(example_audio_config)
example_inputs = (jnp.ones((1, 7, 16)), jnp.ones((1, 7), dtype=bool))
example_params = example_model.init(jax.random.key(0), *example_inputs)
example_outputs = example_model.apply(example_params, *example_inputs)
print('AudioBlock: (1, 7, 16) ->', example_outputs.shape)


# %%
class AudioEncoder(nn.Module):
    config: dict

    @nn.compact
    def __call__(self, features, valid):
        valid = valid.astype(jnp.bool_)
        # Consume audio features: (batch, frames, mel_bins).
        # @arch eg.audio.input:start
        x = features[..., None]
        # @arch eg.audio.input:end
        # Subsample time and frequency twice, then project into audio hidden width.
        # @arch eg.audio.subsample:start
        for index, channels in enumerate(self.config['channels']):
            x = x * valid[:, :, None, None]
            x = nn.Conv(channels, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)),
                        use_bias=False, name=f'convs_{index}')(x)
            normalized = nn.LayerNorm(epsilon=1e-6, use_bias=False, name=f'norms_{index}')(x)
            x = jax.nn.relu(normalized)
            valid = valid[:, ::2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = nn.Dense(self.config['width'], use_bias=False, name='input_projection')(x)
        # @arch eg.audio.subsample:end
        # Encode subsampled audio with relative-attention Conformer blocks.
        # @arch eg.audio.layers:start
        for index in range(self.config['layers']):
            x = AudioBlock(self.config, name=f'layers_{index}')(x, valid)
        # @arch eg.audio.layers:end
        # Project audio hidden states into modality output width: (batch, soft_tokens, audio_output).
        # @arch eg.audio.output:start
        return nn.Dense(self.config['output'], use_bias=True, name='output_projection')(x), valid
        # @arch eg.audio.output:end


# %% [notebook-only]
example_model = AudioEncoder(example_audio_config)
example_inputs = (jnp.ones((1, 25, 8)), jnp.ones((1, 25), dtype=bool))
example_params = example_model.init(jax.random.key(0), *example_inputs)
example_outputs, example_mask = example_model.apply(example_params, *example_inputs)
print('AudioEncoder: (1, 25, 8) ->', example_outputs.shape, example_mask.shape)


# %%
class ModalityProjection(nn.Module):
    input_width: int
    text_width: int

    @nn.compact
    def __call__(self, x):
        # Normalize without a learned scale and project soft tokens to text width.
        # @arch eg.modality.project:start
        normalized = RMSNorm(self.input_width, scale=False, name='norm')(x)
        return nn.Dense(self.text_width, use_bias=False, name='projection')(normalized)
        # @arch eg.modality.project:end


# %% [notebook-only]
example_model = ModalityProjection(24, 16)
example_inputs = jnp.ones((1, 7, 24))
example_params = example_model.init(jax.random.key(0), example_inputs)
example_outputs = example_model.apply(example_params, example_inputs)
print('ModalityProjection: (1, 7, 24) ->', example_outputs.shape)


# %%
def merge_soft_tokens(embeddings, input_ids, token_id, soft_tokens, valid):
    slots = input_ids == token_id
    counts_match = jnp.all(jnp.sum(slots, axis=1) == jnp.sum(valid, axis=1))
    # Eager calls fail clearly; compiled calls return NaN for invalid placeholder counts.
    if not isinstance(counts_match, jax.core.Tracer) and not bool(counts_match):
        raise ValueError('Each example must have one placeholder per valid soft token.')
    feature_order = jnp.argsort(~valid, axis=1, stable=True)
    packed = jnp.take_along_axis(soft_tokens, feature_order[..., None], axis=1)
    rank = jnp.cumsum(slots, axis=1) - 1
    rank = jnp.maximum(rank, 0)
    gathered = jnp.take_along_axis(packed, rank[..., None], axis=1)
    merged = jnp.where(slots[..., None], gathered, embeddings)
    return jnp.where(counts_match, merged, jnp.nan)


# %%
class EmbeddingGemma2(nn.Module):
    text_config: dict
    vision_config: dict | None = None
    audio_config: dict | None = None

    def setup(self):
        self.text = TextBackbone(self.text_config)
        self.vision = VisionEncoder(self.vision_config) if self.vision_config is not None else None
        self.audio = AudioEncoder(self.audio_config) if self.audio_config is not None else None
        self.embed_vision = ModalityProjection(self.vision_config['width'], self.text_config['width']) if self.vision is not None else None
        self.embed_audio = ModalityProjection(self.audio_config['output'], self.text_config['width']) if self.audio is not None else None

    def __call__(self, input_ids, valid, patches=None, patch_positions=None,
                 features=None, feature_valid=None, output_dim=None, video=False):
        valid = valid.astype(jnp.bool_)
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
        count = jnp.maximum(jnp.sum(valid, axis=1, keepdims=True), 1)
        pooled = jnp.sum(masked, axis=1) / count
        # @arch eg.pool:end
        # Keep leading Matryoshka dimensions, then normalize: (batch, output_dim).
        # @arch eg.output:start
        dimension = token_outputs.shape[-1] if output_dim is None else output_dim
        if dimension <= 0 or dimension > token_outputs.shape[-1]:
            raise ValueError('Output dimension must fit the native embedding width.')
        truncated = pooled[:, :dimension]
        norm = jnp.linalg.norm(truncated, axis=-1, keepdims=True)
        return truncated / jnp.maximum(norm, 1e-12)
        # @arch eg.output:end


# %% [notebook-only]
example_model = EmbeddingGemma2(example_text_config)
example_ids = jnp.array([[1, 2, 3, 0]])
example_params = example_model.init(jax.random.key(0), example_ids, example_ids != 0)
example_outputs = example_model.apply(example_params, example_ids, example_ids != 0)
print('EmbeddingGemma2 text: (1, 4) ->', example_outputs.shape)


# %% [notebook-only]
example_model = EmbeddingGemma2(example_text_config, example_vision_config, example_audio_config)
example_ids = jnp.array([[1, 61, 61, 61, 61, 2, 62, 62, 62, 62, 62, 62, 62, 3, 0]])
example_positions = jnp.array([[[x, y] for y in range(4) for x in range(4)]])
example_inputs = (example_ids, example_ids != 0, jnp.ones((1, 16, 12)) * 0.5,
                  example_positions, jnp.ones((1, 25, 8)), jnp.ones((1, 25), dtype=bool))
example_params = example_model.init(jax.random.key(0), *example_inputs, output_dim=6)
example_outputs = example_model.apply(example_params, *example_inputs, output_dim=6)
print('EmbeddingGemma2 text + image + audio: (1, 15) ->', example_outputs.shape,
      'L2 norm:', jnp.linalg.norm(example_outputs, axis=-1))
