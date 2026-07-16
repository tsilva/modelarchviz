# %%
import jax
import jax.numpy as jnp
from flax import linen as nn


# %%
class TimeEmbedding(nn.Module):
    width: int = 128

    @nn.compact
    def __call__(self, timesteps):
        # Build sinusoidal timestep features.
        half_width = self.width // 2  # scalar
        frequencies = jnp.arange(half_width, dtype=jnp.float32)  # -> (half_width)
        frequencies = frequencies / max(half_width - 1, 1)  # (half_width)
        frequencies = jnp.exp(-jnp.log(10000.0) * frequencies)  # (half_width)
        angles = timesteps.astype(jnp.float32)[:, None] * frequencies[None, :]  # (batch), (half_width) -> (batch, half_width)
        embedding = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)  # (batch, half_width) -> (batch, width)

        # Map sinusoidal features into conditioning vectors.
        embedding = nn.Dense(self.width * 4, name="fc1")(embedding)  # (batch, width) -> (batch, width*4)
        embedding = nn.silu(embedding)  # (batch, width*4)
        embedding = nn.Dense(self.width, name="fc2")(embedding)  # (batch, width*4) -> (batch, width)
        return embedding  # (batch, width)


# %% [notebook-only]
# Create and run timestep embeddings: (4) -> (4, 32).
example_embedding = TimeEmbedding(width=32)
example_timesteps = jnp.array([0, 10, 100, 999])  # -> (4)
example_params = example_embedding.init(jax.random.PRNGKey(0), example_timesteps)
example_outputs = example_embedding.apply(example_params, example_timesteps)  # (4) -> (4, 32)
print("time embedding shape:", example_outputs.shape)

# %%
class ResidualBlock(nn.Module):
    out_channels: int
    time_width: int = 128
    groups: int = 8

    @nn.compact
    def __call__(self, x, time_emb):
        # Apply the first normalized convolution and inject timestep conditioning.
        h = nn.GroupNorm(num_groups=self.groups, name="norm1")(x)  # (batch, height, width, in_channels)
        h = nn.silu(h)  # (batch, height, width, in_channels)
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME", name="conv1")(h)  # (batch, height, width, in_channels) -> (batch, height, width, out_channels)
        time_bias = nn.Dense(self.out_channels, name="time_proj")(nn.silu(time_emb))  # (batch, time_width) -> (batch, out_channels)
        time_bias = time_bias[:, None, None, :]  # (batch, out_channels) -> (batch, 1, 1, out_channels)
        h = h + time_bias  # (batch, height, width, out_channels)

        # Finish the residual path and add the channel-matched skip.
        h = nn.GroupNorm(num_groups=self.groups, name="norm2")(h)  # (batch, height, width, out_channels)
        h = nn.silu(h)  # (batch, height, width, out_channels)
        h = nn.Conv(self.out_channels, (3, 3), padding="SAME", name="conv2")(h)  # (batch, height, width, out_channels)
        if x.shape[-1] != self.out_channels:
            skip = nn.Conv(self.out_channels, (1, 1), name="skip")(x)  # (batch, height, width, in_channels) -> (batch, height, width, out_channels)
        else:
            skip = x  # (batch, height, width, out_channels)
        out = h + skip  # (batch, height, width, out_channels)
        return out  # (batch, height, width, out_channels)


# %% [notebook-only]
# Create and run one residual block: (2, 32, 32, 16), (2, 32) -> (2, 32, 32, 32).
example_block = ResidualBlock(out_channels=32, time_width=32)
example_features = jnp.ones((2, 32, 32, 16))  # -> (2, 32, 32, 16)
example_time = jnp.ones((2, 32))  # -> (2, 32)
example_params = example_block.init(jax.random.PRNGKey(1), example_features, example_time)
example_outputs = example_block.apply(example_params, example_features, example_time)  # (2, 32, 32, 16), (2, 32) -> (2, 32, 32, 32)
print("residual output shape:", example_outputs.shape)

# %%
class Downsample(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        # Use a stride-2 convolution for spatial downsampling.
        out = nn.Conv(self.channels, (3, 3), strides=(2, 2), padding="SAME", name="conv")(x)  # (batch, height, width, channels) -> (batch, height/2, width/2, channels)
        return out  # (batch, height/2, width/2, channels)


# %% [notebook-only]
# Create and run downsampling: (2, 32, 32, 32) -> (2, 16, 16, 32).
example_downsample = Downsample(channels=32)
example_features = jnp.ones((2, 32, 32, 32))  # -> (2, 32, 32, 32)
example_params = example_downsample.init(jax.random.PRNGKey(2), example_features)
example_outputs = example_downsample.apply(example_params, example_features)  # (2, 32, 32, 32) -> (2, 16, 16, 32)
print("downsampled shape:", example_outputs.shape)

# %%
class Upsample(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        # Resize spatial dimensions by 2 and refine with a convolution.
        resized_shape = (x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3])  # -> (batch, height*2, width*2, channels)
        x = jax.image.resize(x, resized_shape, method="nearest")  # (batch, height, width, channels) -> (batch, height*2, width*2, channels)
        out = nn.Conv(self.channels, (3, 3), padding="SAME", name="conv")(x)  # (batch, height*2, width*2, channels)
        return out  # (batch, height*2, width*2, channels)


# %% [notebook-only]
# Create and run upsampling: (2, 16, 16, 32) -> (2, 32, 32, 32).
example_upsample = Upsample(channels=32)
example_features = jnp.ones((2, 16, 16, 32))  # -> (2, 16, 16, 32)
example_params = example_upsample.init(jax.random.PRNGKey(3), example_features)
example_outputs = example_upsample.apply(example_params, example_features)  # (2, 16, 16, 32) -> (2, 32, 32, 32)
print("upsampled shape:", example_outputs.shape)

# %%
class UNetDenoiser(nn.Module):
    image_channels: int = 3
    base_channels: int = 64
    time_width: int = 128

    @nn.compact
    def __call__(self, noisy_images, timesteps):
        # Embed timesteps once and run the encoder path.
        time_emb = TimeEmbedding(self.time_width)(timesteps)  # (batch) -> (batch, time_width)
        x = nn.Conv(self.base_channels, (3, 3), padding="SAME", name="input_conv")(noisy_images)  # (batch, height, width, channels) -> (batch, height, width, base)
        skip1 = ResidualBlock(self.base_channels, self.time_width)(x, time_emb)  # (batch, height, width, base)
        x = Downsample(self.base_channels)(skip1)  # (batch, height, width, base) -> (batch, height/2, width/2, base)
        skip2 = ResidualBlock(self.base_channels * 2, self.time_width)(x, time_emb)  # (batch, height/2, width/2, base) -> (batch, height/2, width/2, base*2)
        x = Downsample(self.base_channels * 2)(skip2)  # (batch, height/2, width/2, base*2) -> (batch, height/4, width/4, base*2)

        # Denoise at the bottleneck and decode with U-Net skip concatenations.
        x = ResidualBlock(self.base_channels * 4, self.time_width)(x, time_emb)  # (batch, height/4, width/4, base*2) -> (batch, height/4, width/4, base*4)
        x = Upsample(self.base_channels * 4)(x)  # (batch, height/4, width/4, base*4) -> (batch, height/2, width/2, base*4)
        x = jnp.concatenate([x, skip2], axis=-1)  # (batch, height/2, width/2, base*4) -> (batch, height/2, width/2, base*6)
        x = ResidualBlock(self.base_channels * 2, self.time_width)(x, time_emb)  # (batch, height/2, width/2, base*6) -> (batch, height/2, width/2, base*2)
        x = Upsample(self.base_channels * 2)(x)  # (batch, height/2, width/2, base*2) -> (batch, height, width, base*2)
        x = jnp.concatenate([x, skip1], axis=-1)  # (batch, height, width, base*2) -> (batch, height, width, base*3)
        x = ResidualBlock(self.base_channels, self.time_width)(x, time_emb)  # (batch, height, width, base*3) -> (batch, height, width, base)

        # Predict the Gaussian noise added to the clean image.
        x = nn.GroupNorm(num_groups=8, name="out_norm")(x)  # (batch, height, width, base)
        x = nn.silu(x)  # (batch, height, width, base)
        predicted_noise = nn.Conv(self.image_channels, (3, 3), padding="SAME", name="out_conv")(x)  # (batch, height, width, base) -> (batch, height, width, channels)
        return predicted_noise  # (batch, height, width, channels)


# %% [notebook-only]
# Create and run a small denoising U-Net: (2, 32, 32, 3), (2) -> (2, 32, 32, 3).
example_model = UNetDenoiser(image_channels=3, base_channels=16, time_width=32)
example_noisy = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
example_timesteps = jnp.array([5, 25])  # -> (2)
example_params = example_model.init(jax.random.PRNGKey(4), example_noisy, example_timesteps)
example_outputs = example_model.apply(example_params, example_noisy, example_timesteps)  # (2, 32, 32, 3), (2) -> (2, 32, 32, 3)
print("predicted noise shape:", example_outputs.shape)

# %%
class DDPM(nn.Module):
    image_channels: int = 3
    base_channels: int = 64
    time_width: int = 128
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def schedule(self):
        # Build the fixed forward-process schedule.
        betas = jnp.linspace(self.beta_start, self.beta_end, self.timesteps)  # -> (timesteps)
        alphas = 1.0 - betas  # (timesteps)
        alphas_cumprod = jnp.cumprod(alphas, axis=0)  # (timesteps)
        alphas_cumprod_prev = jnp.concatenate([jnp.ones((1,)), alphas_cumprod[:-1]], axis=0)  # (timesteps)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)  # (timesteps)
        return betas, alphas, alphas_cumprod, posterior_variance

    def extract(self, values, timesteps, target_shape):
        # Gather per-example schedule values and reshape for image broadcasting.
        gathered = values[timesteps]  # (timesteps), (batch) -> (batch)
        broadcast_shape = (timesteps.shape[0],) + (1,) * (len(target_shape) - 1)  # -> (batch, 1, 1, 1)
        gathered = gathered.reshape(broadcast_shape)  # (batch) -> (batch, 1, 1, 1)
        return gathered

    def q_sample(self, clean_images, timesteps, noise):
        # Add scheduled Gaussian noise to clean images in one closed-form step.
        _, _, alphas_cumprod, _ = self.schedule()
        sqrt_alpha = self.extract(jnp.sqrt(alphas_cumprod), timesteps, clean_images.shape)  # (batch, 1, 1, 1)
        sqrt_one_minus_alpha = self.extract(jnp.sqrt(1.0 - alphas_cumprod), timesteps, clean_images.shape)  # (batch, 1, 1, 1)
        noisy_images = sqrt_alpha * clean_images + sqrt_one_minus_alpha * noise  # (batch, height, width, channels)
        return noisy_images  # (batch, height, width, channels)

    @nn.compact
    def __call__(self, noisy_images, timesteps):
        predicted_noise = UNetDenoiser(self.image_channels, self.base_channels, self.time_width)(noisy_images, timesteps)  # (batch, height, width, channels), (batch) -> (batch, height, width, channels)
        return predicted_noise  # (batch, height, width, channels)

    def p_mean_variance(self, params, noisy_images, timesteps):
        # Convert predicted noise into the reverse-process Gaussian mean.
        predicted_noise = self.apply(params, noisy_images, timesteps)  # (batch, height, width, channels)
        betas, alphas, alphas_cumprod, posterior_variance = self.schedule()
        betas_t = self.extract(betas, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        sqrt_one_minus_alpha = self.extract(jnp.sqrt(1.0 - alphas_cumprod), timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        sqrt_recip_alpha = self.extract(jnp.sqrt(1.0 / alphas), timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        model_mean = sqrt_recip_alpha * (noisy_images - betas_t * predicted_noise / sqrt_one_minus_alpha)  # (batch, height, width, channels)
        variance = self.extract(posterior_variance, timesteps, noisy_images.shape)  # (batch, 1, 1, 1)
        return model_mean, variance  # (batch, height, width, channels), (batch, 1, 1, 1)

    def p_sample(self, params, noisy_images, timesteps, noise):
        # Sample one reverse diffusion step.
        model_mean, variance = self.p_mean_variance(params, noisy_images, timesteps)  # (batch, height, width, channels), (batch, 1, 1, 1)
        nonzero_mask = (timesteps != 0).astype(jnp.float32)[:, None, None, None]  # (batch) -> (batch, 1, 1, 1)
        sample = model_mean + nonzero_mask * jnp.sqrt(variance) * noise  # (batch, height, width, channels)
        return sample  # (batch, height, width, channels)


# %% [notebook-only]
# Create and run a small DDPM forward and reverse step.
example_model = DDPM(image_channels=3, base_channels=16, time_width=32, timesteps=50)
example_clean = jnp.ones((2, 32, 32, 3))  # -> (2, 32, 32, 3)
example_noise = jnp.ones_like(example_clean)  # -> (2, 32, 32, 3)
example_timesteps = jnp.array([10, 25])  # -> (2)
example_noisy = example_model.q_sample(example_clean, example_timesteps, example_noise)  # (2, 32, 32, 3)
example_params = example_model.init(jax.random.PRNGKey(5), example_noisy, example_timesteps)
example_predicted_noise = example_model.apply(example_params, example_noisy, example_timesteps)  # (2, 32, 32, 3), (2) -> (2, 32, 32, 3)
example_previous = example_model.p_sample(example_params, example_noisy, example_timesteps, example_noise)  # (2, 32, 32, 3)
print("predicted noise shape:", example_predicted_noise.shape, "sample shape:", example_previous.shape)

# %%
# Train on a tiny synthetic denoising batch.
model = DDPM(image_channels=3, base_channels=16, time_width=32, timesteps=50)
clean_images = jnp.zeros((2, 32, 32, 3))  # -> (2, 32, 32, 3)
clean_images = clean_images.at[0, 4:16, 4:16, :].set(1.0)  # (2, 32, 32, 3)
clean_images = clean_images.at[1, 16:28, 16:28, :].set(1.0)  # (2, 32, 32, 3)
timesteps = jnp.array([10, 25])  # -> (2)
noise = jnp.ones_like(clean_images)  # -> (2, 32, 32, 3)
noisy_images = model.q_sample(clean_images, timesteps, noise)  # (2, 32, 32, 3)
params = model.init(jax.random.PRNGKey(6), noisy_images, timesteps)


def train_step(params, clean_images, timesteps, noise, learning_rate=0.01):
    def loss_fn(current_params):
        noisy_images = model.q_sample(clean_images, timesteps, noise)  # (2, 32, 32, 3)
        predicted_noise = model.apply(current_params, noisy_images, timesteps)  # (2, 32, 32, 3), (2) -> (2, 32, 32, 3)
        loss = jnp.mean((predicted_noise - noise) ** 2)  # (2, 32, 32, 3) -> scalar
        return loss  # scalar

    loss, grads = jax.value_and_grad(loss_fn)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return params, loss


# Fit the model to predict the exact noise added by q_sample.
for step in range(3):
    params, loss = train_step(params, clean_images, timesteps, noise)

# Keep the final scalar loss for inspection.
final_loss = loss  # scalar
